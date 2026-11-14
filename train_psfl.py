"""Single-device prototype-guided split federated patch classification."""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from constellation_generation.generate_orbit_data import detect_contacts, load_setup
from constellation_generation.orbit_model import SPEED_OF_LIGHT_KM_S
from model.patch_classifier import PatchClassifier, build_encoders


ROOT = Path(__file__).resolve().parent


def paired_windows(contacts):
    """Merge overlapping direct contacts for the two satellites in each plane."""
    windows = []
    for plane in sorted({row["plane"] for row in contacts}):
        current = None
        rows = sorted((r for r in contacts if r["plane"] == plane), key=lambda r: r["start_offset_s"])
        for row in rows:
            if current is None or row["start_offset_s"] > current["end_offset_s"]:
                current = {key: row[key] for key in (
                    "plane", "start_offset_s", "end_offset_s", "start_utc", "end_utc",
                    "min_slant_range_km", "uplink_mbps", "downlink_mbps",
                )}
                current["direct_satellites"] = row["satellite_id"]
                windows.append(current)
            else:
                if row["end_offset_s"] > current["end_offset_s"]:
                    current["end_offset_s"] = row["end_offset_s"]
                    current["end_utc"] = row["end_utc"]
                current["min_slant_range_km"] = min(current["min_slant_range_km"], row["min_slant_range_km"])
                direct = set(current["direct_satellites"].split(";"))
                current["direct_satellites"] = ";".join(sorted(direct | {row["satellite_id"]}))
    return sorted(windows, key=lambda r: (r["start_offset_s"], r["plane"]))


def snapshot(modules):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]


def restore(modules, states):
    for module, state in zip(modules, states):
        module.load_state_dict(state)


def average(contributions):
    return [
        {key: torch.stack([state[key] for state in group]).mean(0)
         if value.is_floating_point() else value.clone()
         for key, value in group[0].items()}
        for group in zip(*contributions)
    ]


class PrototypeBank:
    def __init__(self, num_classes, dim, device, momentum):
        self.centers = torch.zeros(num_classes, dim, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, fused, labels):
        features = F.normalize(fused.mean(1), dim=-1)
        for cls in labels.unique().tolist():
            center = features[labels == cls].mean(0)
            if center.norm() <= 1e-6:
                continue
            center = F.normalize(center, dim=0)
            old = self.centers[cls]
            if old.norm() > 1e-6:
                center = self.momentum * old + (1 - self.momentum) * center
            old.copy_(F.normalize(center, dim=0))


def prototype_loss(tokens, labels, centers, temperature):
    if centers is None:
        return tokens.new_zeros(())
    active = centers.norm(dim=1) > 1e-6
    valid = active[labels]
    if active.sum() < 2 or not valid.any():
        return tokens.new_zeros(())
    logits = F.normalize(tokens.mean(1)[valid], dim=-1) @ centers.T / temperature
    return F.cross_entropy(logits.masked_fill(~active, float("-inf")), labels[valid])


@dataclass
class Pair:
    states: list
    batches: object
    clock: float = 0.0
    version: int = 0
    prototypes: torch.Tensor | None = None
    packets: list = field(default_factory=list)


def batches(dataset, indices, batch_size, epochs, seed):
    rng = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=rng)]
        for group in order.split(batch_size):
            optical, radar, labels = zip(*(dataset[i] for i in group.tolist()))
            yield torch.stack(radar), torch.stack(optical), torch.stack(labels)


def offline(pair, plane, stop, workers, bias, training, version, device, log):
    step = max(training["radar_local_compute_s"], training["optical_local_compute_s"])
    if pair.clock + step > stop:
        return
    restore(workers, pair.states)
    mode = training["satellite_encoder_training_mode"]
    for encoder in workers[:2]:
        encoder.requires_grad_(mode == "full")
        if mode == "staged" and version >= training["pretrained_encoder_warmup_aggregations"]:
            count = training["pretrained_encoder_trainable_blocks"]
            if count > 0:
                for block in encoder.transformer.layers[-count:]:
                    block.requires_grad_(True)
    opts = [
        torch.optim.AdamW([
            {"params": enc.parameters(), "lr": training["encoder_learning_rate"]},
            {"params": head.parameters(), "lr": training["auxiliary_learning_rate"]},
        ], weight_decay=training.get("weight_decay", 0.01))
        for enc, head in zip(workers[:2], workers[2:])
    ]
    centers = pair.prototypes.to(device) if pair.prototypes is not None else None
    for _ in range(training["local_steps_per_disconnection"]):
        if pair.clock + step > stop:
            break
        batch = next(pair.batches, None)
        if batch is None:
            break
        radar, optical, labels = (value.to(device) for value in batch)
        row = {"plane": plane, "start_offset_s": pair.clock, "model_version": pair.version}
        features = []
        for name, data, encoder, head, opt in zip(
            ("radar", "optical"), (radar, optical), workers[:2], workers[2:], opts,
        ):
            opt.zero_grad(set_to_none=True)
            tokens = encoder(data, bias, mask_info=None)
            ce = F.cross_entropy(head(tokens), labels)
            proto = prototype_loss(tokens, labels, centers, training["prototype_temperature"])
            (ce + training["prototype_loss_weight"] * proto).backward()
            opt.step()
            features.append(tokens.detach().cpu())
            row[f"{name}_classification_loss"] = ce.item()
            row[f"{name}_prototype_loss"] = proto.item()
        pair.packets.append((*features, labels.cpu()))
        del pair.packets[:-training["recent_smashed_batches"]]
        pair.clock += step
        log.append(row)
    pair.states = snapshot(workers)


def nbytes(t):
    return t.numel() * t.element_size()


def transaction(pair, row, bank, config, aggregate, sync):
    training, link = config["training"], config["link"]
    model_sizes = [sum(nbytes(t) for t in state.values()) for state in pair.states]
    sizes = [model_sizes[i] + model_sizes[i + 2] + link["protocol_overhead_bytes"] for i in range(2)]
    proto = nbytes(bank.centers) if bank else 0
    up = [0, 0] if sync else [
        sizes[i] + sum(nbytes(p[i]) + (nbytes(p[2]) if i == 0 else 0) for p in pair.packets)
        for i in range(2)
    ]
    down = [size + proto for size in sizes]
    combine = max if training["paired_uplink_mode"] == "parallel_full_rate" else sum
    duration = combine(up) * 8 / (row["uplink_mbps"] * 1e6)
    duration += combine(down) * 8 / (row["downlink_mbps"] * 1e6)
    duration += 2 * row["min_slant_range_km"] / SPEED_OF_LIGHT_KM_S
    if not sync:
        duration += len(pair.packets) * training["server_compute_s_per_batch"]
        duration += training["aggregation_compute_s"] if aggregate else 0
    return duration, up, proto


@torch.no_grad()
def evaluate(loader, workers, cross, head, bias, device, classes):
    modules = [*workers[:2], cross, head]
    for module in modules:
        module.eval()
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for optical, radar, labels in loader:
        r = workers[0](radar.to(device), bias, mask_info=None)
        o = workers[1](optical.to(device), bias, mask_info=None)
        pred = head(cross(r, o, bias)).argmax(1).cpu()
        confusion += torch.bincount(labels * classes + pred, minlength=classes**2).reshape(classes, classes)
    for module in modules:
        module.train()
    tp = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - tp
    return (tp.sum() / confusion.sum()).item(), (tp[union > 0] / union[union > 0]).mean().item()


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def run(config, train, test, windows, out):
    t = config["training"]
    torch.manual_seed(t["seed"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" else t["device"])
    if min(t["batch_size"], t["epochs"], t["recent_smashed_batches"]) < 1:
        raise ValueError("batch_size, epochs and recent_smashed_batches must be positive")
    if not (t["prototype_temperature"] > 0 and t["prototype_loss_weight"] >= 0 and 0 <= t["prototype_momentum"] < 1):
        raise ValueError("Invalid prototype temperature, weight or momentum")
    if t["satellite_encoder_training_mode"] not in {"full", "frozen", "staged"}:
        raise ValueError("Invalid satellite_encoder_training_mode")
    if not len(train) or not len(test):
        raise ValueError("Training and test datasets must both be nonempty")
    classes = int(train.targets.max()) + 1
    if int(test.targets.max()) >= classes:
        raise ValueError("Test labels exceed the class range of training labels")
    optical, radar, _ = train[0]
    r, o, cross, bias = build_encoders(config["croma"], radar.shape[-1], radar.shape[0], optical.shape[0], device, ROOT)
    dim = config["croma"]["encoder_dim"]
    workers = [r, o, PatchClassifier(dim, classes).to(device), PatchClassifier(dim, classes).to(device)]
    ground = PatchClassifier(dim, classes).to(device)
    opt = torch.optim.AdamW([*cross.parameters(), *ground.parameters()], lr=t["server_learning_rate"])
    bank = PrototypeBank(classes, dim, device, t["prototype_momentum"]) if t["prototype_learning_enabled"] else None
    global_states = snapshot(workers)
    planes = list(range(1, config["constellation"]["planes"] + 1))
    if not windows or not 1 <= t["aggregation_k"] <= len(planes):
        raise ValueError("Need contact windows and a valid aggregation_k")
    partitions = torch.randperm(len(train), generator=torch.Generator().manual_seed(t["seed"])).tensor_split(len(planes))
    pairs = {
        plane: Pair(global_states, batches(train, ids, t["batch_size"], t["epochs"], t["seed"] + plane))
        for plane, ids in zip(planes, partitions)
    }
    loader = torch.utils.data.DataLoader(test, batch_size=t["batch_size"])
    local_log, contact_log = [], []
    pending = {}
    version, available = 0, 0.0
    for row in windows:
        plane = row["plane"]
        pair = pairs[plane]
        offline(pair, plane, row["start_offset_s"], workers, bias, t, version, device, local_log)
        sync = version - pair.version > t["max_global_version_lag"]
        aggregate = len(pending.keys() | {plane}) >= t["aggregation_k"]
        duration, up, proto_bytes = transaction(pair, row, bank, config, aggregate, sync)
        start = max(available, row["start_offset_s"])
        finish = start + duration
        losses, accuracy, miou = [], "", ""
        status = "completed"
        if not pair.packets and not sync:
            status = "no_matched_features"
        elif finish > row["end_offset_s"] - config["link"]["safety_margin_s"]:
            status = "transaction_does_not_fit_contact"
        else:
            if sync:
                status = "model_synced"
            else:
                for radar_tokens, optical_tokens, labels in pair.packets:
                    opt.zero_grad(set_to_none=True)
                    fused = cross(radar_tokens.to(device), optical_tokens.to(device), bias)
                    labels = labels.to(device)
                    loss = F.cross_entropy(ground(fused), labels)
                    loss.backward()
                    opt.step()
                    if bank:
                        bank.update(fused, labels)
                    losses.append(loss.item())
                pending[plane] = pair.states
                if aggregate:
                    global_states = average(pending.values())
                    pending.clear()
                    version += 1
                    restore(workers, global_states)
                    accuracy, miou = evaluate(loader, workers, cross, ground, bias, device, classes)
                    print(f"[aggregation {version}] end_s={row['end_offset_s']:.1f} "
                          f"server_mean_loss={sum(losses)/len(losses):.4f} "
                          f"test_accuracy={accuracy:.4f} test_miou={miou:.4f}", flush=True)
            pair.states = global_states
            pair.version = version
            pair.prototypes = bank.centers.detach().cpu().clone() if bank else None
            pair.packets.clear()
            available = finish
        pair.clock = max(pair.clock, row["end_offset_s"])
        sent = status in {"completed", "model_synced"}
        contact_log.append({
            "plane": plane, "contact_start_s": row["start_offset_s"], "contact_end_s": row["end_offset_s"],
            "status": status, "transaction_finish_s": finish if sent else "",
            "radar_upload_bytes": up[0] if sent else 0, "optical_upload_bytes": up[1] if sent else 0,
            "prototype_downlink_bytes_per_satellite": proto_bytes if sent else 0,
            "prototype_active_classes": int((pair.prototypes.norm(dim=1) > 1e-6).sum())
            if sent and pair.prototypes is not None else "",
            "global_version": version, "server_updates": len(losses),
            "server_mean_loss": sum(losses) / len(losses) if losses else "",
            "test_accuracy": accuracy, "test_miou": miou,
        })
    horizon = config["simulation"]["duration_hours"] * 3600
    for plane, pair in pairs.items():
        offline(pair, plane, horizon, workers, bias, t, version, device, local_log)
    restore(workers, global_states)
    accuracy, miou = evaluate(loader, workers, cross, ground, bias, device, classes)
    write_csv(out / "local_training_log.csv", sorted(local_log, key=lambda row: row["start_offset_s"]))
    write_csv(out / "training_log.csv", contact_log)
    summary = {"test_accuracy": accuracy, "test_miou": miou, "aggregations": version,
               "server_updates": sum(row["server_updates"] for row in contact_log),
               "train_samples": len(train), "test_samples": len(test),
               "samples_per_plane": {p: len(ids) for p, ids in zip(planes, partitions)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({
        "global_states": dict(zip(("radar_encoder", "optical_encoder", "radar_head", "optical_head"), global_states)),
        "cross_encoder_state": snapshot([cross])[0], "ground_head_state": snapshot([ground])[0],
        "prototype_bank": bank.centers.cpu() if bank else None, "config": config,
    }, out / "checkpoint.pt")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    from datasets.get_dataset import Houston2013Dataset

    config, epoch, constellation, station = load_setup(args.config)
    if config["constellation"]["total_satellites"] != 2 * config["constellation"]["planes"]:
        raise ValueError("PSFL requires two satellites per plane")
    dataset = config["dataset"]
    root = dataset.get("root_dir")
    if root is not None:
        root = ROOT / Path(root).expanduser()
    train, test = (Houston2013Dataset(dataset["patch_size"], split, root) for split in ("train", "test"))
    windows = paired_windows(detect_contacts(config, epoch, constellation, station))
    out = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    (out / "config.json").write_bytes(args.config.read_bytes())
    write_csv(out / "pair_contact_windows.csv", windows)
    run(config, train, test, windows, out)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
