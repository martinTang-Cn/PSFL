# PSFL：卫星多模态 patch 分类

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python train_psfl.py
```

默认使用 Houston2013 的 TRLabel / TSLabel 构造训练集和测试集。HSI 接光学分支，LiDAR 接雷达分支。训练样本以固定随机种子均分给各轨道面，同一对卫星使用相同的样本索引。

- `dataset.patch_size` 为数据裁剪边长，必须为正奇数；`croma.patch_size` 是 ViT 图像块边长，必须整除前者。默认 9×9 patch 产生 81 个 token。
- 每个样本只有一个从 0 开始的中心像素类别。星上辅助头和地面分类头均使用 token mean pooling → Conv1d → GELU → Linear。
- 地面站按类别统计池化的融合特征并以动量更新类别原型。每对卫星在成功接触后取得独立副本；断连时用余弦原型分类损失辅助训练编码器。少于两个有效类别原型时跳过原型损失。
- 保留星地接触窗口、等权聚合和同轨两星可共同接入的假设。未加入断连 ISL 一致性、服务器梯度回传或 EMA teacher。
- `training.prototype_learning_enabled=false` 可关闭原型训练与下发。

训练自动计算接触窗口。结果保存在 `runs/<timestamp>/`，包括配置、窗口、本地损失、地面训练日志和 checkpoint。OA（`test_accuracy`）和 `test_miou` 都根据中心像素分类的混淆矩阵计算。

如需单独导出轨道状态与直连窗口：

```powershell
python -m constellation_generation.generate_orbit_data --config config.json --output-dir outputs
```
