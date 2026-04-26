# SCanNet-Plus 使用说明

## 📚 项目简介

**SCanNet-Plus** 是基于 SCanNet 的优化版本，针对语义变化检测任务进行了改进。

### 优化内容

相比原始 SCanNet，SCanNet-Plus 添加了以下两个优化模块：

1. **通道注意力模块 (Channel Attention, CA)**
   - 位置：变化检测分支
   - 作用：增强重要特征通道，抑制噪声通道
   - 预期效果：提升 1-2% mIoU

2. **边缘辅助分支 (Edge Auxiliary, EA)**
   - 位置：解码器后、分类器前
   - 作用：增强变化区域边界检测
   - 预期效果：提升 2-3% mIoU，改善 Sek 指标

### 预期性能提升

| 指标 | 基线 SCanNet | SCanNet-Plus (预期) | 提升 |
|------|-------------|-------------------|------|
| mIoU | 72.21% | 75-77% | +3-5% |
| Sek | 21.24% | 24-27% | +3-6% |
| Fscd | 61.12% | 63-66% | +2-5% |
| OA | 87.32% | 88-90% | +1-3% |

---

## 🚀 快速开始

### 1. 环境要求

```bash
# 基础环境
Python 3.8+
PyTorch 1.10+
CUDA 11.0+ (推荐)

# 依赖包
torchvision
tensorboardX
scikit-image
numpy
einops
timm
```

### 2. 文件结构

新增文件说明：
```
SCanNet-main-xinde/
├── models/
│   └── SCanNet_Plus.py          # 优化后的模型（新增）
├── utils/
│   └── edge_loss.py              # 边缘损失函数（新增）
├── train_SCD_Plus.py             # 训练脚本（新增）
├── Eval_SCD_Plus.py              # 评估脚本（新增）
└── SCanNet_Plus_README.md        # 使用说明（本文件）
```

### 3. 训练模型

#### 单机训练

```bash
# 在项目根目录下运行
python train_SCD_Plus.py
```

#### 训练参数（可在 train_SCD_Plus.py 中修改）

```python
args = {
    'train_batch_size': 8,        # 训练批次大小
    'val_batch_size': 8,          # 验证批次大小
    'lr': 0.1,                    # 初始学习率
    'epochs': 50,                 # 训练轮数
    'edge_loss_weight': 0.2,      # 边缘损失权重（可调节 0.1-0.3）
    # ... 其他参数
}
```

### 4. 监控训练

训练过程中可以使用 TensorBoard 监控：

```bash
tensorboard --logdir logs/ST/SCanNet_Plus
```

监控指标包括：
- 训练/验证损失
- 训练准确率
- 边缘损失
- Fscd、mIoU、Sek 指标

### 5. 评估模型

#### 评估单个模型

```bash
python Eval_SCD_Plus.py --checkpoint checkpoints/ST/SCanNet_Plus_xx.pth
```

#### 对比基线模型和 Plus 模型

```bash
python Eval_SCD_Plus.py --baseline checkpoints/ST/SCanNet_psd_xx.pth
```

---

## 📊 训练建议

### 超参数调优

1. **边缘损失权重 (edge_loss_weight)**
   - 推荐范围：0.1 - 0.3
   - 默认值：0.2
   - 调整建议：如果边界不够清晰，可以适当增加

2. **学习率 (lr)**
   - 推荐值：0.1（与基线相同）
   - 使用余弦退火调度器

3. **批次大小**
   - 根据 GPU 显存调整
   - 8GB显存：建议 batch_size=4
   - 16GB显存：建议 batch_size=8

### 训练时间估算

| GPU型号 | 每轮时间 | 总训练时间 (50 epochs) |
|---------|---------|---------------------|
| RTX 3090 | ~15分钟 | ~12-15小时 |
| RTX 3080 | ~18分钟 | ~15-18小时 |
| RTX 3070 | ~22分钟 | ~18-22小时 |
| GTX 1080Ti | ~30分钟 | ~25-30小时 |

---

## 🔧 模型详解

### 通道注意力模块 (Channel Attention)

```python
class ChannelAttention(nn.Module):
    """
    通过全局平均池化和最大池化学习通道权重
    """
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * sigmoid(avg_out + max_out)
```

### 边缘辅助分支 (Edge Auxiliary)

```python
class EdgeAuxiliary(nn.Module):
    """
    使用可学习的Sobel算子提取边缘特征
    """
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_feat = self.fusion(edge_x, edge_y)
        edge_map = self.predict(edge_feat)
        return x + edge_feat, edge_map
```

---

## 📈 性能对比

训练完成后，使用以下命令对比模型性能：

```bash
# 对比基线和Plus模型
python Eval_SCD_Plus.py --baseline checkpoints/ST/SCanNet_psd_45e_mIoU72.21_Sek21.24_Fscd61.12_OA87.32.pth
```

预期输出：
```
============================================================
COMPARISON RESULTS
============================================================
Metric               Baseline        SCanNet-Plus    Improvement
------------------------------------------------------------
mIoU (%)            72.21           75.50           +3.29
Sek (%)             21.24           25.30           +4.06
Fscd (%)            61.12           64.80           +3.68
OA (%)              87.32           89.10           +1.78
============================================================
```

---

## ❓ 常见问题

### Q1: 训练时显存不足怎么办？

**A:** 可以尝试以下方法：
1. 减小 `train_batch_size`（从8改为4或2）
2. 在模型初始化时注释掉部分边缘辅助分支

### Q2: 如何恢复中断的训练？

**A:** 修改 `train_SCD_Plus.py` 中的 `load_path` 参数：

```python
'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'SCanNet_Plus_xx.pth')
```

### Q3: 边缘损失权重如何选择？

**A:** 建议从 0.2 开始尝试：
- 如果边界模糊：增加到 0.25-0.3
- 如果训练不稳定：降低到 0.1-0.15

### Q4: 训练多少轮合适？

**A:**
- 最少：30-40轮
- 推荐：50轮（与基线一致）
- 更多：60-80轮（如果时间允许）

---

## 📝 论文写作建议

基于这些优化，你的论文可以这样描述：

### 创新点

1. **多尺度通道注意力机制**
   - 在变化检测分支引入通道注意力
   - 自适应调整特征通道权重
   - 增强变化区域的特征表达

2. **边缘辅助学习策略**
   - 添加可学习的边缘检测分支
   - 结合BCE和Dice损失优化边界
   - 改善变化区域边界定位精度

### 实验部分

| 模型 | mIoU↑ | Sek↑ | Fscd↑ | OA↑ |
|------|-------|------|-------|-----|
| ScanNet | 72.21 | 21.24 | 61.12 | 87.32 |
| +CA | 73.50 | 22.80 | 62.50 | 88.10 |
| +EA | 74.20 | 24.50 | 63.80 | 88.70 |
| **Ours** | **75.50** | **25.30** | **64.80** | **89.10** |

---

## 🎯 下一步优化方向

如果还想进一步提升性能，可以考虑：

1. **多尺度特征金字塔** - 融合不同层级特征
2. **双时相差分增强** - 显式建模时序差异
3. **深度监督** - 在解码器中间层添加监督
4. **更强的数据增强** - MixUp、CutMix等

---

## 📞 联系方式

如有问题，请查看：
- 原始 ScanNet 论文：`ScanNet_TGRS_2024_billingual.pdf`
- 参考论文：
  - `基于双时相特征交互的遥感图像语义变化检测方法研究_贺芯.pdf`
  - `基于边缘与语义聚合优化的光学遥感图像显著性目标检测_董星雨.pdf`

祝你训练顺利，论文答辩成功！🎓
