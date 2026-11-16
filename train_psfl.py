"""Single-device prototype-guided split federated patch classification."""

from __future__ import annotations

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


MODALITIES = ("radar", "optical")


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
    def update(self, tokens, labels):
        features = F.normalize(tokens.mean(1), dim=-1)
        for cls in labels.unique().tolist():
            center = features[labels == cls].mean(0)
            if center.norm() <= 1e-6:
                continue
            center = F.normalize(center, dim=0)
            old = self.centers[cls]
            if old.norm() > 1e-6:
                center = self.momentum * old + (1 - self.momentum) * center
            old.copy_(F.normalize(center, dim=0))


def cross_modal_centers(banks):
    radar, optical = (banks[m].centers for m in MODALITIES)
    shared = (radar.norm(dim=1) > 1e-6) & (optical.norm(dim=1) > 1e-6)
    return ((radar + optical) / 2).masked_fill(~shared[:, None], 0)


def prototype_loss(tokens, labels, centers, temperature):
    if centers is None:
        return tokens.new_zeros(())
    active = centers.norm(dim=1) > 1e-6
    valid = active[labels]
    if active.sum() < 2 or not valid.any():
        return tokens.new_zeros(())
    logits = F.normalize(tokens.mean(1)[valid], dim=-1) @ F.normalize(centers, dim=1).T / temperature
    return F.cross_entropy(logits.masked_fill(~active, float("-inf")), labels[valid])


@dataclass
class Satellite:
    states: list
    batches: object
    clock: float = 0.0
    version: int = 0
    prototypes: torch.Tensor | None = None
    packets: dict = field(default_factory=dict)


def batches(dataset, indices, batch_size, epochs, seed, modality):
    """Carry stable dataset indices through each epoch's shuffled batches."""
    rng = torch.Generator().manual_seed(seed)
    batch_id = 0
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=rng)]
        for group in order.split(batch_size):
            optical, radar, labels = zip(*(dataset[i] for i in group.tolist()))
            yield batch_id, group.tolist(), torch.stack(radar if modality == "radar" else optical), torch.stack(labels)
            batch_id += 1


def offline(sat, plane, modality, stop, workers, bias, training, device, log):
    step = training[f"{modality}_local_compute_s"]
    if sat.clock + step > stop:
        return
    restore(workers, sat.states)
    encoder, head = workers
    mode = training["satellite_encoder_training_mode"]
    encoder.requires_grad_(mode == "full")
    if mode == "staged" and sat.version >= training["pretrained_encoder_warmup_aggregations"]:
        count = training["pretrained_encoder_trainable_blocks"]
        if count > 0:
            for block in encoder.transformer.layers[-count:]:
                block.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": training["encoder_learning_rate"]},
        {"params": head.parameters(), "lr": training["auxiliary_learning_rate"]},
    ], weight_decay=training.get("weight_decay", 0.01))
    centers = sat.prototypes.to(device) if sat.prototypes is not None else None
    for _ in range(training["local_steps_per_disconnection"]):
        if sat.clock + step > stop:
            break
        batch = next(sat.batches, None)
        if batch is None:
            break
        batch_id, sample_ids, data, labels = batch
        opt.zero_grad(set_to_none=True)
        tokens = encoder(data.to(device), bias, mask_info=None)
        labels_device = labels.to(device)
        ce = F.cross_entropy(head(tokens), labels_device)
        proto = prototype_loss(tokens, labels_device, centers, training["prototype_temperature"])
        (ce + training["prototype_loss_weight"] * proto).backward()
        opt.step()
        sat.packets[batch_id] = (sample_ids, tokens.detach().cpu(), labels)
        prune(sat.packets, training["recent_smashed_batches"])
        log.append({"plane": plane, "modality": modality, "batch_id": batch_id,
                    "sample_ids": ";".join(map(str, sample_ids)),
                    "start_offset_s": sat.clock, "model_version": sat.version,
                    "classification_loss": ce.item(), "prototype_loss": proto.item()})
        sat.clock += step
    sat.states = snapshot(workers)


def prune(packets, limit):
    while len(packets) > limit:
        del packets[next(iter(packets))]


def nbytes(t):
    return t.numel() * t.element_size()


def latest_samples(packets):
    """Keep only the latest observation per ID among this satellite's pending batches."""
    samples = {}
    for ids, tokens, labels in packets.values():
        for sample_id, token, label in zip(ids, tokens, labels):
            samples.pop(sample_id, None)
            samples[sample_id] = (token, label)
    return samples


def update_cache(cache, samples, limit):
    for sample_id, (token, label) in samples.items():
        # Refresh replacement order, and avoid retaining the entire batch storage.
        cache.pop(sample_id, None)
        cache[sample_id] = (token.clone(), label.clone())
    prune(cache, limit)


def transaction(sat, row, centers, config, aggregate, sync, samples, matched_batches):
    training, link = config["training"], config["link"]
    size = sum(nbytes(t) for state in sat.states for t in state.values()) + link["protocol_overhead_bytes"]
    proto = nbytes(centers) if centers is not None else 0
    # Include one 64-bit sample ID per uploaded feature.
    up = 0 if sync else size + sum(nbytes(x) + nbytes(y) + 8 for x, y in samples.values())
    down = size + proto
    duration = up * 8 / (row["uplink_mbps"] * 1e6)
    duration += down * 8 / (row["downlink_mbps"] * 1e6)
    duration += 2 * row["min_slant_range_km"] / SPEED_OF_LIGHT_KM_S
    if not sync:
        duration += matched_batches * training["server_compute_s_per_batch"]
        duration += training["aggregation_compute_s"] if aggregate else 0
    return duration, up, down, proto


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


def sample_indices(dataset, count, seed, split):
    if count is None:
        return torch.arange(len(dataset))
    if type(count) is not int or not 1 <= count <= len(dataset):
        raise ValueError(f"dataset.{split}_samples must be null or an integer in [1, {len(dataset)}]")
    return torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]


def run(config, train, test, windows, out):
    t = config["training"]
    torch.manual_seed(t["seed"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" else t["device"])
    cache_limit = t.get("server_feature_cache_samples",
                        t.get("server_feature_cache_batches", t["recent_smashed_batches"]) * t["batch_size"])
    if min(t["batch_size"], t["epochs"], t["recent_smashed_batches"], cache_limit) < 1:
        raise ValueError("Batch size, epochs and feature cache limits must be positive")
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
    warmup_rounds = t.get("prototype_warmup_aggregations", 1)
    if type(warmup_rounds) is not int or warmup_rounds < 0:
        raise ValueError("prototype_warmup_aggregations must be a nonnegative integer")
    banks = {
        m: PrototypeBank(classes, dim, device, t["prototype_momentum"]) for m in MODALITIES
    } if t["prototype_learning_enabled"] else {}
    centers = cross_modal_centers(banks) if banks else None
    global_states = snapshot(workers)
    by_modality = {m: [workers[i], workers[i + 2]] for i, m in enumerate(MODALITIES)}
    global_models = {m: [global_states[i], global_states[i + 2]] for i, m in enumerate(MODALITIES)}
    planes = list(range(1, config["constellation"]["planes"] + 1))
    if not windows or not 1 <= t["aggregation_k"] <= len(planes):
        raise ValueError("Need contact windows and a valid aggregation_k")
    dataset_cfg = config.get("dataset", {})
    train_ids = sample_indices(train, dataset_cfg.get("train_samples"), t["seed"], "train")
    test_ids = sample_indices(test, dataset_cfg.get("test_samples"), t["seed"] + 1, "test")
    if dataset_cfg.get("train_samples") is None:
        train_ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(t["seed"]))
    partitions = train_ids.tensor_split(len(planes))
    satellites = {
        (plane, m): Satellite(global_models[m], batches(train, ids, t["batch_size"], t["epochs"], t["seed"] + plane, m))
        for plane, ids in zip(planes, partitions)
        for m in MODALITIES
    }
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test, test_ids.tolist()), batch_size=t["batch_size"]
    )
    local_log, contact_log = [], []
    pending = {m: {} for m in MODALITIES}
    versions = dict.fromkeys(MODALITIES, 0)
    received = {key: {} for key in satellites}
    available = 0.0
    for row in sorted(windows, key=lambda r: (r["start_offset_s"], r["plane"], r["slot"])):
        # Switch only at a transaction boundary, once both modalities are warmed up.
        stage_two = bool(banks) and min(versions.values()) >= warmup_rounds
        plane = row["plane"]
        modality = MODALITIES[row["slot"] - 1]
        other = MODALITIES[1 - MODALITIES.index(modality)]
        sat = satellites[plane, modality]
        own_cache, peer_cache = received[plane, modality], received[plane, other]
        offline(sat, plane, modality, row["start_offset_s"], by_modality[modality], bias, t, device, local_log)
        sync = versions[modality] - sat.version > t["max_global_version_lag"]
        samples = latest_samples(sat.packets) if not sync else {}
        matched = [sample_id for sample_id in samples if sample_id in peer_cache]
        matched_batches = (len(matched) + t["batch_size"] - 1) // t["batch_size"]
        aggregate = len(pending[modality].keys() | {plane}) >= t["aggregation_k"] and not sync
        duration, up, down, proto_bytes = transaction(
            sat, row, centers if stage_two else None, config, aggregate, sync, samples, matched_batches,
        )
        start = max(available, row["start_offset_s"])
        finish = start + duration
        losses, accuracy, miou = [], "", ""
        status = "completed"
        if not sat.packets and not sync:
            status = "no_local_features"
        elif finish > row["end_offset_s"] - config["link"]["safety_margin_s"]:
            status = "transaction_does_not_fit_contact"
        else:
            if sync:
                status = "model_synced"
            else:
                for start_idx in range(0, len(matched), t["batch_size"]):
                    ids = matched[start_idx:start_idx + t["batch_size"]]
                    local, peer = zip(*((samples[i], peer_cache[i]) for i in ids))
                    radar, optical = (local, peer) if modality == "radar" else (peer, local)
                    radar_tokens, labels = map(torch.stack, zip(*radar))
                    optical_tokens, optical_labels = map(torch.stack, zip(*optical))
                    if not torch.equal(labels, optical_labels):
                        raise RuntimeError(f"Misaligned labels for plane {plane}, samples {ids}")
                    opt.zero_grad(set_to_none=True)
                    fused = cross(radar_tokens.to(device), optical_tokens.to(device), bias)
                    labels = labels.to(device)
                    loss = F.cross_entropy(ground(fused), labels)
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                if stage_two:
                    # Count fresh uploads once, even if the peer has not passed yet.
                    entries = list(samples.values())
                    for group in range(0, len(entries), t["batch_size"]):
                        tokens, labels = map(torch.stack, zip(*entries[group:group + t["batch_size"]]))
                        banks[modality].update(tokens.to(device), labels.to(device))
                    centers = cross_modal_centers(banks)
                update_cache(own_cache, samples, cache_limit)
                if not matched:
                    status = "uploaded_waiting_for_peer"
                pending[modality][plane] = sat.states
                if aggregate:
                    global_models[modality] = average(pending[modality].values())
                    pending[modality].clear()
                    versions[modality] += 1
                    for m in MODALITIES:
                        restore(by_modality[m], global_models[m])
                    accuracy, miou = evaluate(loader, workers, cross, ground, bias, device, classes)
                    mean_loss = f"{sum(losses)/len(losses):.4f}" if losses else "n/a"
                    print(f"[aggregation {modality}:{versions[modality]}] end_s={row['end_offset_s']:.1f} "
                          f"server_mean_loss={mean_loss} "
                          f"test_accuracy={accuracy:.4f} test_miou={miou:.4f}", flush=True)
            sat.states = global_models[modality]
            sat.version = versions[modality]
            sat.prototypes = centers.detach().cpu().clone() if stage_two else None
            sat.packets.clear()
            available = finish
        sat.clock = max(sat.clock, row["end_offset_s"])
        sent = status in {"completed", "model_synced", "uploaded_waiting_for_peer"}
        contact_log.append({
            "plane": plane, "contact_start_s": row["start_offset_s"], "contact_end_s": row["end_offset_s"],
            "satellite_id": row["satellite_id"], "modality": modality,
            "training_phase": 2 if stage_two else 1,
            "status": status, "transaction_finish_s": finish if sent else "",
            "radar_upload_bytes": up if sent and modality == "radar" else 0,
            "optical_upload_bytes": up if sent and modality == "optical" else 0,
            "download_bytes": down if sent else 0,
            "prototype_downlink_bytes_per_satellite": proto_bytes if sent else 0,
            "prototype_active_classes": int((sat.prototypes.norm(dim=1) > 1e-6).sum())
            if sent and sat.prototypes is not None else "",
            "global_version": versions[modality], "server_updates": len(losses),
            "uploaded_samples": len(samples) if sent else 0,
            "matched_samples": len(matched) if sent else 0,
            "cached_samples_for_modality": len(own_cache),
            "server_mean_loss": sum(losses) / len(losses) if losses else "",
            "test_accuracy": accuracy, "test_miou": miou,
        })
    horizon = config["simulation"]["duration_hours"] * 3600
    for (plane, modality), sat in satellites.items():
        offline(sat, plane, modality, horizon, by_modality[modality], bias, t, device, local_log)
    for m in MODALITIES:
        restore(by_modality[m], global_models[m])
    accuracy, miou = evaluate(loader, workers, cross, ground, bias, device, classes)
    write_csv(out / "local_training_log.csv", sorted(local_log, key=lambda row: row["start_offset_s"]))
    write_csv(out / "training_log.csv", contact_log)
    summary = {"test_accuracy": accuracy, "test_miou": miou, "aggregations": sum(versions.values()),
               "aggregations_by_modality": versions,
               "server_updates": sum(row["server_updates"] for row in contact_log),
               "train_samples": len(train_ids), "test_samples": len(test_ids),
               "samples_per_plane": {p: len(ids) for p, ids in zip(planes, partitions)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({
        "global_states": {f"{m}_{name}": state for m in MODALITIES
                          for name, state in zip(("encoder", "head"), global_models[m])},
        "cross_encoder_state": snapshot([cross])[0], "ground_head_state": snapshot([ground])[0],
        "prototype_bank": centers.cpu() if centers is not None else None,
        "modality_prototype_banks": {m: bank.centers.cpu() for m, bank in banks.items()},
        "config": config,
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
    windows = detect_contacts(config, epoch, constellation, station)
    out = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    (out / "config.json").write_bytes(args.config.read_bytes())
    write_csv(out / "contact_windows.csv", windows)
    run(config, train, test, windows, out)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
