# PSFL：卫星多模态 patch 分类

在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python train_psfl.py
```

默认使用 Houston2013 的 TRLabel / TSLabel 构造训练集和测试集。HSI 接光学分支，LiDAR 接雷达分支。训练样本以固定随机种子均分给各轨道面，同一对卫星使用相同的样本索引。

- `dataset.train_samples` 和 `dataset.test_samples` 默认分别为 400 和 1000，各自在原始训练集和测试集内按种子无放回抽样。设为 `null` 使用对应的完整数据集。整个训练过程使用同一个测试子集。
- `dataset.patch_size` 为数据裁剪边长，必须为正奇数；`croma.patch_size` 是 ViT 图像块边长，必须整除前者。默认 9×9 patch 产生 81 个 token。
- 每个样本只有一个从 0 开始的中心像素类别。星上辅助头和地面分类头均使用 token mean pooling → Conv1d → GELU → Linear。
- 每颗卫星只在自己的直连窗口内上传、下载。窗口重叠时，单链路地面站按窗口开始时间依次安排事务；事务必须在各自的窗口截止前完成。
- 样本编号使用训练集索引，跨 epoch 不变。地面站按轨道面、模态、样本编号缓存最新特征；新上传的样本与另一模态已有的同编号特征配对，再组成融合训练 batch。融合后两份特征仍保留，可在后续有一侧上传更新时再次使用。每个轨道面每种模态最多缓存 `training.server_feature_cache_samples` 条，超过上限时淘汰最久未写入的条目。
- 雷达与光学模型分别在收到 `aggregation_k` 次有效模型上传后等权聚合。同一颗卫星多次成功上传分别计数，跳过或仅下载同步的窗口不计数。两种模态各完成 `training.prototype_warmup_aggregations` 次聚合前，卫星只用分类交叉熵训练，地面站继续融合匹配的特征。达到阈值后，从下一笔星地事务进入第二阶段。
- 第二阶段，地面站用新上传的单模态特征分别更新雷达和光学类别中心。某类别在两种模态中都有中心时，取两者的等权平均作为共享原型。成功过站的卫星收到共享原型副本后，用分类交叉熵加上加权的余弦原型对比损失训练。没有有效共享原型时只用分类交叉熵。
- `training.prototype_learning_enabled=false` 可关闭原型训练与下发。

训练自动计算接触窗口。结果保存在 `runs/<timestamp>/`，包括配置、窗口、本地损失、地面训练日志和 checkpoint。OA（`test_accuracy`）和 `test_miou` 都根据中心像素分类的混淆矩阵计算。

如需单独导出轨道状态与直连窗口：

```powershell
python -m constellation_generation.generate_orbit_data --config config.json --output-dir outputs
```
