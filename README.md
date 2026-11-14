# PSFL：卫星多模态 patch 分类

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python train_psfl.py
```

默认使用 Houston2013 的 TRLabel / TSLabel 构造训练集和测试集。HSI 接光学分支，LiDAR 接雷达分支。训练样本以固定随机种子均分给各轨道面，同一对卫星使用相同的样本索引。

- `dataset.patch_size` 为数据裁剪边长，必须为正奇数；`croma.patch_size` 是 ViT 图像块边长，必须整除前者。默认 9×9 patch 产生 81 个 token。
- 每个样本只有一个从 0 开始的中心像素类别。星上辅助头和地面分类头均使用 token mean pooling → Conv1d → GELU → Linear。
- 每颗卫星只在自己的直连窗口内上传、下载。窗口重叠时，单链路地面站按窗口开始时间依次安排事务；事务必须在各自的窗口截止前完成。
- 样本编号使用训练集索引，跨 epoch 不变。地面站按轨道面、模态、样本编号缓存最新特征；新上传的样本与另一模态已有的同编号特征配对，再组成融合训练 batch。融合后两份特征仍保留，可在后续有一侧上传更新时再次使用。每个轨道面每种模态最多缓存 `training.server_feature_cache_samples` 条，超过上限时淘汰最久未写入的条目。
- 地面站按融合特征更新类别原型。只有成功过站的卫星才下载自己的原型副本。雷达与光学模型分别在收齐 `aggregation_k` 个轨道面的贡献后等权聚合。
- `training.prototype_learning_enabled=false` 可关闭原型训练与下发。

训练自动计算接触窗口。结果保存在 `runs/<timestamp>/`，包括配置、窗口、本地损失、地面训练日志和 checkpoint。OA（`test_accuracy`）和 `test_miou` 都根据中心像素分类的混淆矩阵计算。

如需单独导出轨道状态与直连窗口：

```powershell
python -m constellation_generation.generate_orbit_data --config config.json --output-dir outputs
```
