===============================================================================
                    ACDC 2017 心脏 MRI 分割项目研究报告
===============================================================================

项目概述
---------
本项目基于 ACDC 2017 (Automatic Cardiac Diagnosis Challenge) 数据集，研究并
实现了多种 U-Net 变体架构用于心脏 MRI 图像的左心室(LV)、右心室(RV)和心肌
(Myocardium)的自动分割。项目完成了从数据预处理到模型训练、评估的完整深度学习
实验流程。

===============================================================================
第一部分：深度学习框架与技术栈
===============================================================================

一、核心框架选择

1. 深度学习框架：TensorFlow 2.10 + Keras
   - 版本：tensorflow-gpu 2.10.1
   - 选择理由：
     * Keras 高层 API 便于快速原型开发
     * TensorFlow 生态成熟，医学影像处理支持完善
     * GPU 加速支持（CUDA 11.2 + cuDNN 8.1）

2. 医学图像处理库：Nibabel
   - 用于读取 .nii.gz 格式的医学影像数据
   - 支持 NIfTI (Neuroimaging Informatics Technology Initiative) 格式

3. 分割模型库：segmentation-models 1.0.1
   - 提供成熟的 U-Net 实现及多种预训练骨干网络
   - 支持的骨干网络：VGG16, ResNet, EfficientNet 等
   - 内置 Dice Loss、Focal Loss 等分割专用损失函数

4. 科学计算栈
   - NumPy 1.23.5：数值计算
   - Pandas 2.0.3：数据处理
   - SciPy 1.10.1：Hausdorff 距离计算
   - Scikit-learn 1.7.2：数据集划分

5. 可视化与报告
   - Matplotlib 3.10.8：结果可视化
   - python-docx：自动生成 Word 格式评估报告

二、开发环境配置

- Python 版本：3.10.13
- 环境管理：Conda
- GPU 支持：NVIDIA CUDA 11.2 + cuDNN 8.1
- 日志管理：Loguru

三、项目架构设计

项目采用模块化设计，主要模块如下：

baseline/
├── data_loader/          # 数据加载与预处理模块
│   └── data_load_impl.py # ACDC 数据集处理、标准化、划分
├── train_core/           # 训练核心模块
│   └── train_core.py     # 四种模型训练函数
├── evaluate/             # 评估模块
│   └── evaluate_impl.py  # Dice、HD 等指标计算与可视化
├── base_component/       # 基础组件
│   └── base.py          # U-Net++、Attention U-Net 架构定义
├── log_helpers.py        # 日志重定向工具
├── train_pipeline.py     # 主训练流程入口
└── generate_evaluation_reports.py  # 自动报告生成器

===============================================================================
第二部分：医学影像数据处理
===============================================================================

一、ACDC 2017 数据集介绍

1. 数据集概述
   - 来源：2017 MICCAI 自动心脏诊断挑战赛
   - 任务：心脏 MRI 短轴切面的左心室、右心室、心肌分割
   - 标签定义：
     * 0: 背景 (Background)
     * 1: 右心室 (RV - Right Ventricle)
     * 2: 心肌 (Myocardium)
     * 3: 左心室 (LV - Left Ventricle)

2. 数据特点
   - 3D MRI 体积数据（.nii.gz 格式）
   - 舒张末期和收缩末期两个时间点
   - 包含健康和病理心脏（肥厚型心肌病、扩张型心肌病等）

二、数据预处理流程

数据预处理在 baseline/data_loader/data_load_impl.py 中实现：

1. 强度归一化 (Z-Score Normalization)
   -----------------------------------
   def normalize_volume(img_data):
       p99 = np.percentile(img_data, 99)
       p1 = np.percentile(img_data, 1)
       img_data = np.clip(img_data, p1, p99)
       mean = np.mean(img_data)
       std = np.std(img_data)
       return (img_data - mean) / (std + 1e-8)

   目的：消除不同扫描设备间的强度差异，抑制离群值

2. 空间标准化 (中心填充/裁剪)
   ---------------------------
   def center_padding_or_crop(img, target_size=(256, 256)):
       # 将图像中心对齐，不足部分填充0，超出部分裁剪
       # 统一输出尺寸为 256x256

   目的：统一输入尺寸，便于批量训练

3. 2D 切片展开
   -------------
   - 将 3D 体积数据沿 Z 轴展开为 2D 切片序列
   - 每个切片独立作为训练样本
   - 保留空间信息的同时增加样本数量

4. 数据集划分（按病人级别）
   ------------------------
   from sklearn.model_selection import train_test_split

   train_ids, test_val_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
   val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

   划分比例：训练集 70% / 验证集 15% / 测试集 15%
   重要：按病人而非切片划分，防止数据泄露

三、数据加载优化

1. 内存映射 (Memory Mapping)
   ---------------------------
   x_mmap = np.load(x_npy_path, mmap_mode='r')

   优势：不将全部数据加载到内存，按需读取，支持大规模数据集训练

2. 懒加载 Pipeline (tf.data.Dataset)
   ----------------------------------
   def make_dataset(x_path, y_path, batch_size, num_classes):
       ds = tf.data.Dataset.from_generator(
           lambda: np_generator(x_path, y_path, num_classes),
           output_signature=output_signature
       )
       ds = ds.batch(batch_size)
       ds = ds.prefetch(tf.data.AUTOTUNE)
       return ds

   优势：GPU 训练与 CPU 数据预处理并行，提高训练效率

3. One-Hot 编码
   -------------
   将标签编码为 4 通道 one-hot 格式：
   - Channel 0: label == 1 (RV)
   - Channel 1: label == 2 (Myocardium)
   - Channel 2: label == 3 (LV)
   - Channel 3: label == 0 (Background)

===============================================================================
第三部分：U-Net 变体架构实现
===============================================================================

本项目实现并对比了四种 U-Net 变体架构：

一、VGG16-UNet (Baseline)

1. 架构特点
   ----------
   - 使用 segmentation_models 库的标准实现
   - 编码器：VGG16（ImageNet 预训练）
   - 解码器：对称上采样路径
   - 跳跃连接：直接连接编码器和解码器同层特征

2. 网络配置
   ----------
   backbone_name = "vgg16"
   encoder_weights = "imagenet"  # 关键：使用预训练权重
   input_shape = (256, 256, 3)
   classes = 4
   activation = "softmax"

3. 模型参数
   ----------
   总参数量：23,752,714 (~23.8M)
   编码器起点通道数：64 (VGG16 第一层)

二、Lightweight U-Net (轻量化 U-Net)

1. 架构设计
   ----------
   标准对称 U-Net 架构，无预训练权重：

   编码器（下采样路径）：
   - Block 1: 32 filters, 256x256 -> 128x128
   - Block 2: 64 filters, 128x128 -> 64x64
   - Block 3: 128 filters, 64x64 -> 32x32
   - Block 4: 256 filters, 32x32 -> 16x16
   - Bottleneck: 512 filters, 16x16

   解码器（上采样路径）：
   - 对称结构，每层上采样后与编码器对应层拼接
   - 渐进式 Dropout: 0.1 -> 0.2 -> 0.3

2. 卷积块设计
   ------------
   def conv_block(inputs, filters, dropout_rate=0.1):
       x = Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal')(inputs)
       x = BatchNormalization()(x)
       x = Activation('relu')(x)
       x = Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal')(x)
       x = BatchNormalization()(x)
       x = Activation('relu')(x)
       if dropout_rate > 0:
           x = Dropout(dropout_rate)(x)
       return x

3. 模型参数
   ----------
   总参数量：7,752,714 (~7.8M)
   约为 Baseline 的 33%

三、U-Net++ (嵌套跳跃连接)

1. 核心创新
   ----------
   - 密集跳跃连接：解码器接收来自多个编码器层的特征
   - 嵌套上采样路径：多层特征逐步融合
   - 深层监督（本项目未实现）

2. 架构示意
   ----------
   nb_filter = [32, 64, 128, 256, 512]

   编码器第一列：
   X0_0 -> X1_0 -> X2_0 -> X3_0 -> X4_0

   嵌套层：
   X0_1 = up(X1_0) + X0_0
   X0_2 = up(X1_1) + X0_1 + X0_0
   X0_3 = up(X1_2) + X0_2 + X0_1 + X0_0
   X0_4 = up(X1_3) + X0_3 + X0_2 + X0_1 + X0_0  # 输出层

3. 实现要点
   ----------
   def standard_unit(input_tensor, stage, col, filters):
       x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
       x = BatchNormalization()(x)
       x = Activation('relu')(x)
       x = Conv2D(filters, (3, 3), padding='same')(x)
       x = BatchNormalization()(x)
       x = Activation('relu')(x)
       return x

4. 模型参数
   ----------
   总参数量：9,170,148 (~9.2M)
   约为 Baseline 的 38%

四、Attention U-Net (注意力门控)

1. 核心创新
   ----------
   - 注意力门控机制：自适应加权跳跃连接特征
   - 抑制无关区域，聚焦目标结构
   - Additive Attention 实现

2. 注意力门控设计
   ----------------
   def attention_gate(x, g, inter_shape):
       # x: 编码器特征 (skip connection)
       # g: 解码器门控信号 (gating signal)

       theta_x = Conv2D(inter_shape, (1, 1), padding='same')(x)
       phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)

       f = Add()([theta_x, phi_g])
       f = Activation('relu')(f)

       psi_f = Conv2D(1, (1, 1), padding='same')(f)
       sigmoid_psi = Activation('sigmoid')(psi_f)

       return multiply([x, sigmoid_psi])  # 加权原始特征

3. 架构特点
   ----------
   - 编码器与 Lightweight U-Net 相同
   - 解码器每层使用注意力门控替代直接拼接
   - 注意力权重范围：[0, 1]，自动学习关注区域

4. 模型参数
   ----------
   总参数量：8,120,520 (~8.1M)
   约为 Baseline 的 34%

五、架构对比总结

| 模型 | 参数量 | 预训练 | 起点通道 | 核心创新 |
|------|--------|--------|----------|----------|
| VGG16-UNet | 23.8M | ImageNet | 64 | 成熟库+预训练 |
| Lightweight U-Net | 7.8M | 无 | 32 | 轻量化设计 |
| U-Net++ | 9.2M | 无 | 32 | 密集跳跃连接 |
| Attention U-Net | 8.1M | 无 | 32 | 注意力门控 |

===============================================================================
第四部分：研究方案设计
===============================================================================

一、损失函数设计

1. 组合损失函数
   -------------
   采用 Dice Loss 与 Categorical Focal Loss 的组合：

   dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
   focal_loss = sm.losses.CategoricalFocalLoss()
   total_loss = dice_loss + focal_loss

   类别权重配置：[RV=1.0, Myo=1.0, LV=1.0, BG=0.5]
   降低背景类权重，关注解剖结构

2. Dice Loss
   -----------
   - 定义：Dice = 2*|X∩Y| / (|X| + |Y|)
   - 优势：直接优化分割重叠度，对类别不平衡不敏感
   - 实现：segmentation_models 库内置

3. Categorical Focal Loss
   ------------------------
   - 定义：FL(p) = -α(1-p)^γ log(p)
   - 优势：聚焦难分样本，减少简单样本主导
   - γ (focusing parameter)：默认 2.0

二、评价指标体系

1. 主要指标
   ----------
   a) Dice 系数 (F1 Score for segmentation)
      - 定义：2TP / (2TP + FP + FN)
      - 范围：[0, 1]，越大越好
      - 分别计算 RV、Myocardium、LV 三个类别

   b) Hausdorff Distance (HD)
      - 定义：两个点集间的最大最小距离
      - 公式：HD(A,B) = max(h(A,B), h(B,A))
               其中 h(A,B) = max(a∈A) min(b∈B) ||a-b||
      - 实现使用：scipy.spatial.distance.directed_hausdorff
      - 反映边界分割精度，越小越好

2. 辅助指标
   ----------
   - IOU (Intersection over Union)
   - Accuracy (准确率)
   - Recall (召回率/敏感度)
   - Specificity (特异性)
   - F1-Score

3. 评价实现
   ----------
   def calculate_dice(y_true, y_pred):
       for i in range(1, 4):  # RV, Myo, LV
           gt = (y_true == i).astype(np.uint8)
           pred = (y_pred == i).astype(np.uint8)
           intersection = np.sum(gt * pred)
           sum_val = np.sum(gt) + np.sum(pred)
           dice = (2. * intersection / sum_val) if sum_val > 0 else 1.0

   def calculate_hausdorff(y_true, y_pred):
       for i in range(1, 4):
           gt_points = np.argwhere(y_true == i)
           pred_points = np.argwhere(y_pred == i)
           d1 = directed_hausdorff(gt_points, pred_points)[0]
           d2 = directed_hausdorff(pred_points, gt_points)[0]
           hd = max(d1, d2)

三、训练配置

1. 优化器
   --------
   optimizer = Adam(learning_rate=1e-4)

2. 训练参数
   ----------
   - Batch Size: 4
   - Epochs: 60
   - 输入尺寸: 256x256x1 (单通道)
   - GPU 内存增长: 开启（防 OOM）

3. 监控指标
   ----------
   metrics = [
       sm.metrics.IOUScore(threshold=0.5),
       sm.metrics.FScore(threshold=0.5)
   ]

四、实验结果 (60 Epochs)

| 模型 | 参数量 | RV Dice | Myo Dice | LV Dice | RV HD | Myo HD | LV HD |
|------|--------|---------|----------|---------|-------|--------|-------|
| VGG16-UNet | 23.8M | 0.779 | 0.877 | 0.943 | 15.53 | 2.77 | 2.17 |
| Lightweight U-Net | 7.8M | 0.746 | 0.844 | 0.913 | 15.14 | 4.23 | 3.15 |
| U-Net++ | 9.2M | 0.742 | 0.817 | 0.872 | 14.94 | 7.64 | 8.68 |
| Attention U-Net | 8.1M | 0.725 | 0.809 | 0.880 | 16.33 | 7.14 | 6.83 |

五、关键技术总结

1. 预训练权重的决定性作用
   -----------------------
   VGG16-UNet 使用 ImageNet 预训练权重，性能显著优于其他从零训练的模型，
   证明在医学影像分割任务中，自然图像预训练特征仍有重要价值。

2. 模型容量与性能的关系
   ---------------------
   U-Net++ 和 Attention U-Net 参数量约为 Baseline 的 1/3，性能下降明显，
   说明复杂架构需要足够的网络宽度才能发挥优势。

3. 数据集划分策略
   ---------------
   按病人级别划分数据集，避免同一病人的不同切片同时出现在训练和测试集，
   确保评估结果的真实泛化能力。

4. 内存优化技术
   -------------
   使用 numpy mmap 和 tf.data.Dataset 实现流式数据加载，避免将全部数据
   加载到内存，使得在有限 GPU 内存下也能进行大规模数据训练。
