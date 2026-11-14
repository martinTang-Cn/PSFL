import numpy as np
import random
from pathlib import Path
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA


def _standardize_houston_modalities(hsi, lidar):
    hsi = hsi.astype(np.float32)
    scaler = StandardScaler()
    for band in range(hsi.shape[2]):
        hsi[:, :, band] = scaler.fit_transform(hsi[:, :, band])
    lidar = StandardScaler().fit_transform(lidar.astype(np.float32))[..., np.newaxis]
    return hsi, lidar


def _load_houston_2012():
    root = "./data/12Houston-17Houston/2012Houston"
    hsi = sio.loadmat(f"{root}/HSI_data.mat")["HSI_data"]
    lidar = sio.loadmat(f"{root}/LiDAR_data.mat")["LiDAR_data"]
    labels = sio.loadmat(f"{root}/All_Label.mat")["All_Label"]
    hsi, lidar = _standardize_houston_modalities(hsi, lidar)
    return hsi, lidar, labels


class MyDataset(Dataset):
    """端到端数据集：构造时即在类内部完成 mat 读取、标准化/PCA、patch 切分与标签对齐。

    用法:
        train_set = MyDataset("MUUFL-Trento", client_id=0, patch_size=9, num_samples_per_class=40, split="train")
        test_set = MyDataset("MUUFL-Trento", client_id=0, patch_size=9, num_samples_per_class=40, split="test")
    """

    _raw_cache = {}
    _patch_cache = {}

    def __init__(self, dataset_name, client_id, patch_size=9, num_samples_per_class=40, split="train"):
        super(MyDataset, self).__init__()
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        self.dataset_name = dataset_name
        self.client_id = client_id
        self.patch_size = patch_size
        self.num_samples_per_class = num_samples_per_class
        self.split = split

        prepared = self._prepare(dataset_name, client_id, patch_size, num_samples_per_class)
        self.data = prepared["data"]              # 该 client 的模态数据（已标准化/PCA）
        self.gt = prepared["gt"]                  # 全图标签
        self.TR_map = prepared["TR_map"]          # 采样后的训练标签图
        self.TE_map = prepared["TE_map"]          # 测试标签图
        self.data_patch = prepared["data_patch_train" if split == "train" else "data_patch_test"]
        self.label_map = (self.TR_map if split == "train" else self.TE_map)
        self.label_map = self.label_map[self.label_map != 0]


    def __getitem__(self, idx):
        return ([torch.from_numpy(data[idx]) for data in self.data_patch],
                torch.tensor(self.label_map[idx] - 1).to(torch.long))

    def __len__(self):
        return len(self.label_map)

    ## =================================================================================================================
    @staticmethod
    def apply_pca_to_hsi(hsi_data: np.ndarray, target_dim: int = 63) -> np.ndarray:
        h, w, c = hsi_data.shape
        hsi_reshaped = hsi_data.reshape(-1, c)

        pca = PCA(n_components=target_dim)
        hsi_pca = pca.fit_transform(hsi_reshaped)

        return hsi_pca.reshape(h, w, target_dim)

    ## =================================================================================================================
    @classmethod
    def get_data(cls, dataset_name="MUUFL-Trento"):
        """读取并预处理数据集，返回 [[HSI, LiDAR, gt], ...]，结果按数据集名缓存。"""
        if dataset_name not in cls._raw_cache:
            cls._raw_cache[dataset_name] = cls._load_data(dataset_name)
        return cls._raw_cache[dataset_name]

    @classmethod
    def _load_data(cls, dataset_name="MUUFL-Trento"):
        if dataset_name == "MUUFL-Trento":
            data_HSI_MUUFL = sio.loadmat("./data/Trento-MUUFL/MUUFL/HSI_data.mat")["HSI_data"].astype(np.float32)  #  (325, 220, 64)
            data_LiDAR_MUUFL = sio.loadmat("./data/Trento-MUUFL/MUUFL/LiDAR_data.mat")["LiDAR_data"].astype(np.float32)  # (325, 220, 2)
            gt_MUUFL = sio.loadmat("./data/Trento-MUUFL/MUUFL/All_Label.mat")["All_Label"]

            scaler1 = StandardScaler()  #
            hsi_ = np.zeros_like(data_HSI_MUUFL)
            for i in range(data_HSI_MUUFL.shape[2]):
                hsi_[:, :, i] = scaler1.fit_transform(data_HSI_MUUFL[:, :, i])
            hsi_ = cls.apply_pca_to_hsi(hsi_, target_dim=63)
            data_HSI_MUUFL = hsi_

            scaler2 = StandardScaler()
            lidar_ = scaler2.fit_transform(data_LiDAR_MUUFL[:, :, 0])
            lidar_ = lidar_.reshape(lidar_.shape[0], lidar_.shape[1], -1)
            data_LiDAR_MUUFL = lidar_

            data_HSI_Trento = sio.loadmat("./data/Trento-MUUFL/Trento/HSI_data.mat")["HSI_data"].astype(np.float32)  # (166, 600, 63)
            data_LiDAR_Trento = sio.loadmat("./data/Trento-MUUFL/Trento/LiDAR_data.mat")["LiDAR_data"].astype(np.float32)  # (166, 600)
            gt_Trento = sio.loadmat("./data/Trento-MUUFL/Trento/All_Label.mat")["All_Label"]

            scaler3 = StandardScaler()  #
            hsi_ = np.zeros_like(data_HSI_Trento)
            for i in range(data_HSI_Trento.shape[2]):
                hsi_[:, :, i] = scaler3.fit_transform(data_HSI_Trento[:, :, i])
            data_HSI_Trento = hsi_

            scaler4 = StandardScaler()
            lidar_ = scaler4.fit_transform(data_LiDAR_Trento)
            lidar_ = lidar_.reshape(lidar_.shape[0], lidar_.shape[1], -1)
            data_LiDAR_Trento = lidar_

            return [[data_HSI_MUUFL, data_LiDAR_MUUFL, gt_MUUFL], [data_HSI_MUUFL, data_LiDAR_MUUFL, gt_MUUFL],
                    [data_HSI_Trento, data_LiDAR_Trento, gt_Trento]]

        elif dataset_name == "H13-H18":
            data_HSI_H13, data_LiDAR_H13, gt_H13 = _load_houston_2012()

            data_HSI_H18 = sio.loadmat("./data/12Houston-17Houston/2017Houston/HSI_data.mat")["HSI_data"].astype(np.float32)  # (210, 954, 48)
            data_LiDAR_H18 = sio.loadmat("./data/12Houston-17Houston/2017Houston/LiDAR_data.mat")["LiDAR_data"].astype(np.float32)
            gt_H18 = sio.loadmat("./data/12Houston-17Houston/2017Houston/All_Label.mat")["All_Label"]

            scaler3 = StandardScaler()  #
            hsi_ = np.zeros_like(data_HSI_H18)
            for i in range(data_HSI_H18.shape[2]):
                hsi_[:, :, i] = scaler3.fit_transform(data_HSI_H18[:, :, i])
            data_HSI_H18 = hsi_

            scaler4 = StandardScaler()
            lidar_ = scaler4.fit_transform(data_LiDAR_H18)
            lidar_ = lidar_.reshape(lidar_.shape[0], lidar_.shape[1], -1)
            data_LiDAR_H18 = lidar_

            return [[data_HSI_H13, data_LiDAR_H13, gt_H13], [data_HSI_H13, data_LiDAR_H13, gt_H13],
                    [data_HSI_H18, data_LiDAR_H18, gt_H18]]

        else:
            raise ValueError("Unknown dataset")

    ## ===========================================================================================
    @staticmethod
    def build_patch_data(Data, gt=None, patchsize=9, num_samples_per_class=40, Client_id=None):
        gt_flatten = np.reshape(gt, -1)
        train_mask = np.zeros_like(gt_flatten)
        for j in range(gt.max()):
            pos = np.where(gt_flatten == j + 1)[0]
            pos_id = random.sample(range(len(pos)), num_samples_per_class)
            train_mask[np.array(pos[pos_id])] = 1
        train_mask = np.reshape(train_mask, (gt.shape[0], gt.shape[1]))
        train_mask = train_mask.astype(np.bool_)
        TR_map = np.multiply(train_mask, gt)
        TE_map = gt

        data_pad = []
        for data in Data:
            bands = data.shape[-1]
            temp = data[:, :, 0]
            pad_width = np.floor(patchsize / 2)
            pad_width = np.int32(pad_width)
            temp2 = np.pad(temp, pad_width, 'symmetric')
            [h_pad, w_pad] = temp2.shape
            data_pad_ = np.empty((h_pad, w_pad, bands), dtype='float32')
            for i in range(bands):
                temp = data[:, :, i]
                pad_width = np.floor(patchsize / 2)
                pad_width = np.int32(pad_width)
                temp2 = np.pad(temp, pad_width, 'symmetric')
                data_pad_[:, :, i] = temp2
            data_pad.append(data_pad_)

        data_patch_train = [np.empty((sum(sum(TR_map != 0)), data.shape[-1], patchsize, patchsize), dtype='float32') for data in Data]
        data_patch_test = [np.empty((sum(sum(TE_map != 0)), data.shape[-1], patchsize, patchsize), dtype='float32') for data in Data]

        ind1, ind2 = np.where(TR_map != 0)
        ind3 = ind1 + pad_width
        ind4 = ind2 + pad_width
        for i, data_pad_ in enumerate(data_pad):
            for j in range(len(ind1)):
                patch = data_pad_[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                        (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]

                patch = np.reshape(patch, (patchsize * patchsize, -1))
                patch = np.transpose(patch)
                patch = np.reshape(patch, (-1, patchsize, patchsize))

                data_patch_train[i][j, :, :, :] = patch

        ind1, ind2 = np.where(TE_map != 0)
        ind3 = ind1 + pad_width
        ind4 = ind2 + pad_width
        for i, data_pad_ in enumerate(data_pad):
            for j in range(len(ind1)):
                patch = data_pad_[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                        (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]

                patch = np.reshape(patch, (patchsize * patchsize, -1))
                patch = np.transpose(patch)
                patch = np.reshape(patch, (-1, patchsize, patchsize))
                data_patch_test[i][j, :, :, :] = patch

        if Client_id == 0:
            data_patch_train[0] = np.zeros_like(data_patch_train[0])
            data_patch_test[0] = np.zeros_like(data_patch_test[0])
        elif Client_id == 1:
            data_patch_train[1] = np.zeros_like(data_patch_train[1])
            data_patch_test[1] = np.zeros_like(data_patch_test[1])

        return data_patch_train, data_patch_test, TR_map, TE_map

    ## ===========================================================================================
    @classmethod
    def _prepare(cls, dataset_name, client_id, patch_size, num_samples_per_class):
        """一次准备好某个 client 的全部训练/测试 patch，按参数缓存以便 train/test 共享同一份采样。"""
        key = (dataset_name, client_id, patch_size, num_samples_per_class)
        if key not in cls._patch_cache:
            client_data = cls.get_data(dataset_name=dataset_name)[client_id]
            data, gt = client_data[:-1], client_data[-1]
            data_patch_train, data_patch_test, TR_map, TE_map = cls.build_patch_data(
                data,
                gt=gt,
                patchsize=patch_size,
                num_samples_per_class=num_samples_per_class,
                Client_id=client_id,
            )
            cls._patch_cache[key] = {
                "data": data,
                "gt": gt,
                "TR_map": TR_map,
                "TE_map": TE_map,
                "data_patch_train": data_patch_train,
                "data_patch_test": data_patch_test,
            }
        return cls._patch_cache[key]

    @classmethod
    def clear_cache(cls):
        cls._raw_cache.clear()
        cls._patch_cache.clear()


class Houston2013Dataset(Dataset):
    """返回对齐的 HSI/LiDAR patch 和从 0 开始的中心像素类别。"""

    def __init__(self, patch_size=9, split="train", data_dir=None):
        super().__init__()
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        label_keys = {"train": "TRLabel", "test": "TSLabel"}
        if split not in label_keys:
            raise ValueError("split must be 'train' or 'test'")

        self.patch_size = patch_size
        root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "Houston2013"
        label_key = label_keys[split]
        labels = sio.loadmat(root / f"{label_key}.mat")[label_key].toarray()
        hsi = sio.loadmat(root / "HSI.mat")["HSI"].transpose(1, 2, 0)
        lidar = sio.loadmat(root / "LiDAR_1.mat")["LiDAR_1"][0]
        hsi, lidar = _standardize_houston_modalities(hsi, lidar)
        self._rows, self._cols = np.nonzero(labels)

        pad = self.patch_size // 2
        spatial_padding = ((pad, pad), (pad, pad))
        self._hsi = np.pad(hsi, (*spatial_padding, (0, 0)), mode="symmetric")
        self._lidar = np.pad(lidar, (*spatial_padding, (0, 0)), mode="symmetric")
        self.targets = torch.from_numpy(labels[self._rows, self._cols].astype(np.int64) - 1)

    def __getitem__(self, idx):
        row, col = self._rows[idx], self._cols[idx]
        patch = np.s_[row:row + self.patch_size, col:col + self.patch_size]
        return (
            torch.from_numpy(self._hsi[patch].transpose(2, 0, 1).copy()),
            torch.from_numpy(self._lidar[patch].transpose(2, 0, 1).copy()),
            self.targets[idx],
        )

    def __len__(self):
        return len(self._rows)
