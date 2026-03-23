# 模型性能差距排查报告

**创建时间**: 2026-03-23
**分析对象**: ACDC心脏MRI分割项目 - U-Net变体模型性能对比
**问题**: U-Net++ 和 Attention U-Net 性能低于 Baseline (VGG16-UNet)

---

## 一、实验结果概述

### 1.1 性能对比表 (60 Epochs)

| 模型 | 参数量 | RV Dice | Myo Dice | LV Dice | RV HD | Myo HD | LV HD |
|------|--------|---------|----------|---------|-------|--------|-------|
| **VGG16-UNet (Baseline)** | 23.8M | **0.779** | **0.877** | **0.943** | 15.53 | **2.77** | **2.17** |
| **Lightweight U-Net** | 7.8M | 0.746 | 0.844 | 0.913 | **15.14** | 4.23 | 3.15 |
| **U-Net++** | 9.2M | 0.742 | 0.817 | 0.872 | 14.94 | 7.64 | 8.68 |
| **Attention U-Net** | 8.1M | 0.725 | 0.809 | 0.880 | 16.33 | 7.14 | 6.83 |

### 1.2 关键发现

- **Baseline (VGG16-UNet)** 性能最佳，符合预期（使用了ImageNet预训练权重）
- **Lightweight U-Net** 表现第二好，证明轻量化架构可行
- **U-Net++ 和 Attention U-Net** 性能低于预期，未能超越Baseline
- **边界分割误差**：U-Net++ 的 LV Hausdorff 距离高达 8.68，远高于 Baseline 的 2.17

---

## 二、代码审查结果

### 2.1 模型架构对比

#### VGG16-UNet (Baseline)
**文件位置**: `baseline/train_core/train_core.py:94-110`

```python
base_model = sm.Unet(
    backbone_name="vgg16",
    input_shape=(256, 256, 3),
    classes=4,
    activation="softmax",
    encoder_weights="imagenet"  # 使用ImageNet预训练权重
)
```

**特点**:
- 使用 segmentation_models 库的成熟实现
- VGG16 编码器（ImageNet预训练）
- 参数量: 23,752,714
- 编码器起点通道数: 64 (VGG16第一层)

#### U-Net++ (Custom)
**文件位置**: `baseline/base_component/base.py:36-77`

```python
def build_unet_plus_plus_custom(input_shape=(256, 256, 1), num_classes=4):
    nb_filter = [32, 64, 128, 256, 512]  # 通道数配置
    # ... 密集跳跃连接实现
```

**特点**:
- 手动实现
- 无预训练权重，从零训练
- 参数量: 9,170,148 (仅为Baseline的38%)
- 编码器起点通道数: 32 (比Baseline少一半)

#### Attention U-Net (Custom)
**文件位置**: `baseline/base_component/base.py:104-150`

```python
def build_attention_unet_custom(input_shape=(256, 256, 1), num_classes=4):
    # ... 注意力门控实现
    c1 = conv_block(inputs, 32)   # 起点通道数32
```

**特点**:
- 手动实现
- 无预训练权重
- 参数量: 8,120,520 (仅为Baseline的34%)
- 编码器起点通道数: 32

---

## 三、问题根因分析

### 问题1: 模型容量严重不足 🔴

| 模型 | 参数量 | 相对Baseline | 起点通道数 | 瓶颈通道数 |
|------|--------|-------------|-----------|-----------|
| VGG16-UNet | 23.8M | 100% | 64 | 512 |
| U-Net++ | 9.2M | 38% | 32 | 512 |
| Attention U-Net | 8.1M | 34% | 32 | 512 |

**分析**:
- 第一层通道数决定了模型能捕捉多少低级特征（边缘、纹理）
- Baseline有64个通道，自定义模型只有32个
- U-Net++的优势在于密集跳跃连接和深层特征融合，但如果每层通道数太小，传递的信息量就不足

**代码位置**: `baseline/base_component/base.py:49`
```python
nb_filter = [32, 64, 128, 256, 512]  # 建议改为 [64, 128, 256, 512, 1024]
```

---

### 问题2: 无预训练权重 🔴

| 模型 | 预训练权重 | 训练起点 |
|------|-----------|---------|
| VGG16-UNet | ImageNet | 从高质量特征提取器开始 |
| U-Net++ | 无 | 从随机初始化开始 |
| Attention U-Net | 无 | 从随机初始化开始 |

**分析**:
- ImageNet预训练权重提供了强大的通用视觉特征提取能力
- 医学图像数据量相对较小，从零训练难以达到相同效果
- 这是最可能导致性能差距的核心因素

---

### 问题3: U-Net++架构与原始论文偏差 🟡

**原始论文配置**:
- 通道数: `[64, 128, 256, 512, 1024]`
- 更深、更宽的网络

**当前实现配置**:
```python
nb_filter = [32, 64, 128, 256, 512]  # 通道数减半
```

**影响**:
- 密集跳跃连接的优势无法充分发挥
- 每个连接传递的特征图通道数不足

---

### 问题4: Attention Gate 实现细节 🟡

**文件位置**: `baseline/base_component/base.py:80-102`

```python
def attention_gate(x, g, inter_shape):
    theta_x = Conv2D(inter_shape, (1, 1), padding='same')(x)  # 无kernel_initializer
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    f = Add()([theta_x, phi_g])
    f = Activation('relu')(f)
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    sigmoid_psi = Activation('sigmoid')(psi_f)
    return multiply([x, sigmoid_psi])
```

**潜在问题**:
1. **Conv2D层没有指定初始化**：使用默认Glorot uniform，可能与网络其他部分的he_normal不一致
2. **Sigmoid饱和风险**：注意力权重可能过早饱和，导致梯度消失
3. **与原始论文差异**：原始实现可能有不同的正则化策略

**对比 conv_block 的实现** (`base.py:20`):
```python
x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)  # 有显式初始化
```

---

### 问题5: 超参数未单独调优 🔴

**当前实现**: 所有模型使用相同的超参数

```python
# train_core.py:544-546
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 统一学习率
    loss=dice_loss + focal_loss,
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)
```

**问题**:
- 复杂模型（U-Net++）通常需要更小的学习率
- 没有学习率调度器（ReduceLROnPlateau, CosineDecay等）
- 没有warmup策略
- 60个epoch可能不足以让U-Net++收敛

---

### 问题6: 缺少数据增强 🟡

**文件位置**: `baseline/train_core/train_core.py:36-63`

```python
def np_generator(x_npy_path, y_npy_path, num_classes):
    # 只做了归一化，没有数据增强
    x_slice = x_slice / maxv
    # ... 没有旋转、翻转、形变等增强
```

**原始U-Net论文使用的数据增强**:
- 随机旋转（±15°）
- 随机弹性形变
- 平移、缩放
- 灰度值变化
- 翻转

**影响**:
- 对于复杂模型（U-Net++/AttU-Net），数据量不足时更容易过拟合
- 简单模型（Lightweight U-Net）反而泛化更好

---

### 问题7: Batch Size 与 BatchNormalization 🟡

**当前设置**: `BATCH_SIZE = 4`

**问题**:
- 小batch size导致BatchNormalization统计不稳定
- U-Net++和Attention U-Net使用了BN，但batch size太小
- 可能影响训练稳定性和最终性能

---

### 问题8: Dropout策略不一致 🟡

| 模型 | Dropout策略 |
|------|------------|
| Lightweight U-Net | 0.1 → 0.2 → 0.3 渐进式 |
| U-Net++ | 无Dropout |
| Attention U-Net | 固定0.1 (在conv_block中) |

**文件位置**: `baseline/base_component/base.py:16`
```python
def conv_block(inputs, filters, dropout_rate=0.1):  # 默认0.1
    # ...
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
```

---

## 四、问题优先级总结

| 优先级 | 问题 | 影响程度 | 修复难度 |
|--------|------|----------|----------|
| P0 | 模型容量不足（通道数太小） | 🔴 严重 | 低 |
| P0 | 无预训练权重 | 🔴 严重 | 中 |
| P1 | 超参数未调优（统一学习率） | 🔴 严重 | 低 |
| P1 | 训练不充分（epoch数） | 🟡 中等 | 低 |
| P2 | 无数据增强 | 🟡 中等 | 中 |
| P2 | Attention Gate初始化问题 | 🟡 中等 | 低 |
| P2 | BN与小batch size冲突 | 🟡 中等 | 中 |
| P3 | Dropout策略不一致 | 🟢 轻微 | 低 |

---

## 五、改进建议

### 5.1 快速改进（低投入高回报）

1. **增加模型通道数**
   ```python
   # 修改前
   nb_filter = [32, 64, 128, 256, 512]

   # 修改后
   nb_filter = [64, 128, 256, 512, 1024]
   ```

2. **添加学习率调度器**
   ```python
   from tensorflow.keras.callbacks import ReduceLROnPlateau
   reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
   ```

3. **增加训练轮数**
   ```python
   EPOCHS = 100  # 从60增加到100
   ```

### 5.2 中期改进

1. **添加数据增强**（在数据加载阶段）
2. **修复Attention Gate初始化**
3. **调整Dropout策略**

### 5.3 长期改进

1. **使用更大的预训练编码器**（EfficientNet, ResNet50）
2. **尝试segmentation_models库的官方U-Net++实现**
3. **进行完整的超参数搜索**

---

## 六、论文写作角度

这个"失败"的实验结果对论文也有价值：

### 6.1 可讨论的论点

1. **预训练权重的优势**
   - 证明在医学图像分割任务中，ImageNet预训练权重仍有显著优势
   - 从零训练的复杂模型难以超越预训练的简单模型

2. **模型容量与性能的关系**
   - 参数量少一半，性能显著下降
   - 说明U-Net++等架构需要足够的网络宽度才能发挥优势

3. **复杂度的代价**
   - 更复杂的架构（U-Net++, AttU-Net）不一定带来更好的性能
   - 在数据量有限的情况下，简单模型+预训练权重可能是更好的选择

4. **工程实现的重要性**
   - 理论上的优势需要正确的实现和充分的训练才能体现
   - 超参数调优、数据增强等因素对最终性能影响很大

### 6.2 论文结构建议

```
4. 结果与分析
4.1 性能对比
   - 表格展示所有模型的Dice和Hausdorff指标

4.2 结果分析
   4.2.1 预训练权重的决定性作用
   4.2.2 模型容量对性能的影响
   4.2.3 为什么U-Net++/AttU-Net表现不佳
   4.2.4 轻量化模型的可行性

4.3 讨论
   - 复杂模型的训练挑战
   - 医学图像分割中的数据效率问题
   - 未来改进方向
```

---

## 七、相关文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| 模型定义 | `baseline/base_component/base.py` | U-Net++和Attention U-Net实现 |
| 训练核心 | `baseline/train_core/train_core.py` | 所有模型的训练逻辑 |
| 主训练管道 | `baseline/train_pipeline.py` | 训练流程入口 |
| 评估实现 | `baseline/evaluate/evaluate_impl.py` | 评估指标计算 |
| 结果目录 | `baseline/result/` | 所有实验结果 |
| 原始实现 | `baseline_raw/` | 原始参考代码 |

---

**报告结束**
