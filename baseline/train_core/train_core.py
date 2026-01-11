import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    concatenate, BatchNormalization, Activation, Dropout
)
from loguru import logger

# -------------------------------
# GPU 设置
# -------------------------------
def setup_gpu():
    """防 OOM / GPU 设置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 启用按需增长，避免一次性占满显存导致 OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory growth: {e}")
    else:
        logger.info("No GPU found, using CPU")

# -------------------------------
# 辅助：generator 使用 mmap，按需做标准化与 one-hot
# -------------------------------
def np_generator(x_npy_path, y_npy_path, num_classes):
    """
    从磁盘按样本逐个读取（使用 numpy mmap 模式），返回 (x, y)：
    - x: shape (256,256,1), float32, 已经除以该切片的最大值（若 max=0 则原样不变）
    - y: shape (256,256,4), float32, one-hot 按 (label==1,2,3,0)
    保证：不把全部数据一次性加载到内存。
    """
    x_mmap = np.load(x_npy_path, mmap_mode='r')
    y_mmap = np.load(y_npy_path, mmap_mode='r')
    n_samples = x_mmap.shape[0]
    for i in range(n_samples):
        # x: (256,256,1)
        x_slice = x_mmap[i, :, :, 0].astype(np.float32)
        maxv = np.max(x_slice)
        if maxv > 0:
            x_slice = x_slice / maxv
        x_slice = np.expand_dims(x_slice, axis=-1)

        # y: one-hot 4 channels; 按原逻辑映射：ch0=(==1), ch1=(==2), ch2=(==3), ch3=(==0)
        y_slice_raw = y_mmap[i, :, :, 0]
        h, w = y_slice_raw.shape
        y_onehot = np.zeros((h, w, num_classes), dtype=np.float32)
        y_onehot[:, :, 0] = (y_slice_raw == 1)
        y_onehot[:, :, 1] = (y_slice_raw == 2)
        y_onehot[:, :, 2] = (y_slice_raw == 3)
        y_onehot[:, :, 3] = (y_slice_raw == 0)

        yield x_slice, y_onehot

# -------------------------------
# 构建 tf.data.Dataset（懒加载）
# -------------------------------
def make_dataset(x_path, y_path, batch_size, num_classes):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, num_classes), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: np_generator(x_path, y_path, num_classes),
        output_signature=output_signature
    )
    # 不做 cache（会占内存），不做 shuffle（保持原有顺序/逻辑）
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------------
# 构建模型
# -------------------------------
def build_model(num_classes, backbone, encoder_weights, activation, input_shape):
    """
    构建模型（与原逻辑等价：先用 base_model 接受 3 通道，然后用 Conv2D 将 1->3）
    """
    logger.info(f"Building model with backbone: {backbone}")
    logger.info(f"Encoder weights: {encoder_weights}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Activation: {activation}")
    
    base_model = sm.Unet(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=num_classes,
        activation=activation,
        encoder_weights=encoder_weights
    )

    # 使用任意空间大小输入，但通道数为 1（数据生成器输出）
    inp = Input(shape=(None, None, 1))
    x = Conv2D(3, (1, 1))(inp)  # 将单通道映射到 3 通道，保持原逻辑
    out = base_model(x)
    modelUnet = Model(inp, out, name=base_model.name)
    
    logger.info(f"Model built successfully: {modelUnet.name}")
    return modelUnet

# -------------------------------
# 主训练函数
# -------------------------------
def train_model(save_data_folder, SAVE_BASE_PATH,EPOCHS=50, BATCH_SIZE=4):
    """
    训练 U-Net 分割模型
    
    参数:
        save_data_folder: npy 文件所在文件夹路径
        SAVE_BASE_PATH: 模型保存基础路径
        BATCH_SIZE: 批次大小，默认 4
        EPOCHS: 训练轮数，默认 50
        
    返回:
        model_save_path: 模型保存的完整路径
    """
    # -------------------------------
    # 设置 GPU
    # -------------------------------
    setup_gpu()
    
    # -------------------------------
    # 参数设置
    # -------------------------------
    NUM_CLASSES = 4
    BACKBONE = "vgg16"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = "softmax"
    INPUT_SHAPE = (256, 256, 3)   # base_model 所需的输入 (因为我们会 map 1->3)
    
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    logger.info(f"Backbone: {BACKBONE}")
    logger.info(f"Encoder weights: {ENCODER_WEIGHTS}")
    logger.info(f"Activation: {ACTIVATION}")
    logger.info(f"Data folder: {save_data_folder}")
    logger.info(f"Save base path: {SAVE_BASE_PATH}")
    logger.info("=" * 60)
    
    # 保存路径
    MODEL_SAVE_DIR = os.path.join(SAVE_BASE_PATH, f"risultati_{EPOCHS}_model")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Model save directory created: {MODEL_SAVE_DIR}")
    
    MODEL_SAVE_NAME = os.path.join(MODEL_SAVE_DIR, f"modelUnet_{EPOCHS}epochs.keras")
    logger.info(f"Model will be saved to: {MODEL_SAVE_NAME}")
    
    # -------------------------------
    # 数据路径
    # -------------------------------
    X_TRAIN_PATH = os.path.join(save_data_folder, "x_2d_train.npy")
    Y_TRAIN_PATH = os.path.join(save_data_folder, "y_2d_train.npy")
    X_VAL_PATH = os.path.join(save_data_folder, "x_2d_val.npy")
    Y_VAL_PATH = os.path.join(save_data_folder, "y_2d_val.npy")
    X_TEST_PATH = os.path.join(save_data_folder, "x_2d_test.npy")
    Y_TEST_PATH = os.path.join(save_data_folder, "y_2d_test.npy")
    
    logger.info("Checking data files...")
    for path in [X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        logger.info(f"✓ Found: {path}")
    
    # -------------------------------
    # 创建数据集
    # -------------------------------
    logger.info("Creating training dataset...")
    train_ds = make_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("Creating validation dataset...")
    val_ds = make_dataset(X_VAL_PATH, Y_VAL_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("Creating test dataset...")
    test_ds = make_dataset(X_TEST_PATH, Y_TEST_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("All datasets created successfully")
    
    # -------------------------------
    # 构建模型
    # -------------------------------
    modelUnet = build_model(NUM_CLASSES, BACKBONE, ENCODER_WEIGHTS, ACTIVATION, INPUT_SHAPE)
    
    # -------------------------------
    # 编译模型（保持原损失/指标不变）
    # -------------------------------
    logger.info("Compiling model...")
    lr = 1e-4
    optim = Adam(learning_rate=lr)
    logger.info(f"Optimizer: Adam with learning rate {lr}")
    
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss
    logger.info("Loss function: Dice Loss + Categorical Focal Loss")
    logger.info("Class weights for Dice Loss: [1, 1, 1, 0.5]")
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    logger.info("Metrics: IOU Score (threshold=0.5), F-Score (threshold=0.5)")
    
    modelUnet.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    logger.info("Model compiled successfully")
    
    # -------------------------------
    # 训练（使用 tf.data 流式训练以避免 OOM）
    # -------------------------------
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    history = modelUnet.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        # 若需要更稳健的训练监控，可以添加 callbacks（例如 ModelCheckpoint），
        # 但原始逻辑只在训练结束时保存最终模型，所以这里保持一致。
    )
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    
    # -------------------------------
    # 在 test 集上评估（与原逻辑保持一致：提供测试评估）
    # -------------------------------
    logger.info("Evaluating model on test set...")
    test_metrics = modelUnet.evaluate(test_ds)
    logger.info(f"Test metrics: {test_metrics}")
    
    # -------------------------------
    # 保存模型（与原逻辑相同）
    # -------------------------------
    logger.info(f"Saving model to: {MODEL_SAVE_NAME}")
    modelUnet.save(MODEL_SAVE_NAME)
    logger.info("Model saved successfully!")
    
    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final model location: {MODEL_SAVE_NAME}")
    logger.info("=" * 60)
    
    return MODEL_SAVE_NAME




# -------------------------------
# GPU 设置
# -------------------------------
def setup_gpu():
    """防 OOM / GPU 设置"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory growth: {e}")
    else:
        logger.info("No GPU found, using CPU")

# -------------------------------
# 构建轻量级 U-Net（无预训练骨干）
# -------------------------------
def conv_block(inputs, filters, dropout_rate=0.1):
    """
    基础卷积块：Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU -> Dropout
    """
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    return x

def build_lightweight_unet(input_shape=(256, 256, 1), num_classes=4, activation='softmax'):
    """
    构建轻量级 U-Net 模型（标准 U-Net 架构）
    
    架构设计：
    - 编码器（下采样路径）：4 层，通道数 [32, 64, 128, 256]
    - 瓶颈层：512 通道
    - 解码器（上采样路径）：4 层，对称结构 + skip connections
    - 输出层：使用 softmax 激活进行多类分割
    
    参数：
        input_shape: 输入图像尺寸，默认 (256, 256, 1)
        num_classes: 分类数量，默认 4
        activation: 输出激活函数，默认 'softmax'
    """
    inputs = Input(shape=input_shape)
    
    # ============ 编码器（Encoder / 下采样路径）============
    # Block 1: 256x256 -> 128x128
    conv1 = conv_block(inputs, 32, dropout_rate=0.1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2: 128x128 -> 64x64
    conv2 = conv_block(pool1, 64, dropout_rate=0.1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3: 64x64 -> 32x32
    conv3 = conv_block(pool2, 128, dropout_rate=0.2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4: 32x32 -> 16x16
    conv4 = conv_block(pool3, 256, dropout_rate=0.2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # ============ 瓶颈层（Bottleneck）============
    # 16x16 with 512 filters
    conv5 = conv_block(pool4, 512, dropout_rate=0.3)
    
    # ============ 解码器（Decoder / 上采样路径）============
    # Block 6: 16x16 -> 32x32
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(256, (2, 2), padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)  # Skip connection
    conv6 = conv_block(merge6, 256, dropout_rate=0.2)
    
    # Block 7: 32x32 -> 64x64
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)  # Skip connection
    conv7 = conv_block(merge7, 128, dropout_rate=0.2)
    
    # Block 8: 64x64 -> 128x128
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(64, (2, 2), padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)  # Skip connection
    conv8 = conv_block(merge8, 64, dropout_rate=0.1)
    
    # Block 9: 128x128 -> 256x256
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)  # Skip connection
    conv9 = conv_block(merge9, 32, dropout_rate=0.1)
    
    # ============ 输出层 ============
    outputs = Conv2D(num_classes, (1, 1), activation=activation, name='output')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs, name='Lightweight_UNet')
    
    return model

# -------------------------------
# 主训练函数
# -------------------------------
def train_model_light_uent(save_data_folder, SAVE_BASE_PATH, EPOCHS=50, BATCH_SIZE=4):
    """
    训练轻量级 U-Net 分割模型
    
    参数:
        save_data_folder: npy 文件所在文件夹路径
        SAVE_BASE_PATH: 模型保存基础路径
        BATCH_SIZE: 批次大小，默认 4
        EPOCHS: 训练轮数，默认 50
        
    返回:
        model_save_path: 模型保存的完整路径
    """
    # -------------------------------
    # 设置 GPU
    # -------------------------------
    setup_gpu()
    
    # -------------------------------
    # 参数设置
    # -------------------------------
    NUM_CLASSES = 4
    ACTIVATION = "softmax"
    INPUT_SHAPE = (256, 256, 1)  # 轻量级 U-Net 直接接受单通道输入
    
    logger.info("=" * 60)
    logger.info("Training Configuration - Lightweight U-Net")
    logger.info("=" * 60)
    logger.info(f"Model: Lightweight U-Net (No Pretrained Backbone)")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    logger.info(f"Activation: {ACTIVATION}")
    logger.info(f"Input shape: {INPUT_SHAPE}")
    logger.info(f"Data folder: {save_data_folder}")
    logger.info(f"Save base path: {SAVE_BASE_PATH}")
    logger.info("=" * 60)
    
    # 保存路径
    MODEL_SAVE_DIR = os.path.join(SAVE_BASE_PATH, f"lightweight_unet_{EPOCHS}_model")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    logger.info(f"Model save directory created: {MODEL_SAVE_DIR}")
    
    MODEL_SAVE_NAME = os.path.join(MODEL_SAVE_DIR, f"lightweight_unet_{EPOCHS}epochs.keras")
    logger.info(f"Model will be saved to: {MODEL_SAVE_NAME}")
    
    # -------------------------------
    # 数据路径
    # -------------------------------
    X_TRAIN_PATH = os.path.join(save_data_folder, "x_2d_train.npy")
    Y_TRAIN_PATH = os.path.join(save_data_folder, "y_2d_train.npy")
    X_VAL_PATH = os.path.join(save_data_folder, "x_2d_val.npy")
    Y_VAL_PATH = os.path.join(save_data_folder, "y_2d_val.npy")
    X_TEST_PATH = os.path.join(save_data_folder, "x_2d_test.npy")
    Y_TEST_PATH = os.path.join(save_data_folder, "y_2d_test.npy")
    
    logger.info("Checking data files...")
    for path in [X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH, X_TEST_PATH, Y_TEST_PATH]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        logger.info(f"✓ Found: {path}")
    
    # -------------------------------
    # 创建数据集
    # -------------------------------
    logger.info("Creating training dataset...")
    train_ds = make_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("Creating validation dataset...")
    val_ds = make_dataset(X_VAL_PATH, Y_VAL_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("Creating test dataset...")
    test_ds = make_dataset(X_TEST_PATH, Y_TEST_PATH, BATCH_SIZE, NUM_CLASSES)
    
    logger.info("All datasets created successfully")
    
    # -------------------------------
    # 构建轻量级 U-Net 模型
    # -------------------------------
    logger.info("Building Lightweight U-Net model...")
    modelUnet = build_lightweight_unet(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        activation=ACTIVATION
    )
    
    # 打印模型摘要
    logger.info("Model architecture:")
    modelUnet.summary(print_fn=lambda x: logger.info(x))
    
    # 计算模型参数量
    total_params = modelUnet.count_params()
    logger.info(f"Total parameters: {total_params:,}")
    
    # -------------------------------
    # 编译模型（保持原损失/指标不变）
    # -------------------------------
    logger.info("Compiling model...")
    lr = 1e-4
    optim = Adam(learning_rate=lr)
    logger.info(f"Optimizer: Adam with learning rate {lr}")
    
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss
    logger.info("Loss function: Dice Loss + Categorical Focal Loss")
    logger.info("Class weights for Dice Loss: [1, 1, 1, 0.5]")
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    logger.info("Metrics: IOU Score (threshold=0.5), F-Score (threshold=0.5)")
    
    modelUnet.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    logger.info("Model compiled successfully")
    
    # -------------------------------
    # 训练（使用 tf.data 流式训练以避免 OOM）
    # -------------------------------
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    history = modelUnet.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    
    # -------------------------------
    # 在 test 集上评估
    # -------------------------------
    logger.info("Evaluating model on test set...")
    test_metrics = modelUnet.evaluate(test_ds)
    logger.info(f"Test metrics: {test_metrics}")
    
    # -------------------------------
    # 保存模型
    # -------------------------------
    logger.info(f"Saving model to: {MODEL_SAVE_NAME}")
    modelUnet.save(MODEL_SAVE_NAME)
    logger.info("Model saved successfully!")
    
    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final model location: {MODEL_SAVE_NAME}")
    logger.info("=" * 60)
    
    return MODEL_SAVE_NAME