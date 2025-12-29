import os
from pathlib import Path
import gc
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger
import segmentation_models as sm

# 尝试开启 GPU 内存按需增长（避免一次占满）
def enable_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("GPU config error: %s", e)

enable_gpu_memory_growth()

# 映射 index -> 原始标签值（与原脚本一致）
_IDX_TO_LABEL = np.array([1, 2, 3, 0], dtype=np.int32)

def pred_probs_to_label(pred_probs):
    """pred_probs: (N,H,W,4) -> labels (N,H,W,1) 按映射返回 {0,1,2,3}"""
    idx = np.argmax(pred_probs, axis=-1)   # (N,H,W)
    labels = _IDX_TO_LABEL[idx]            # (N,H,W) values in {0,1,2,3}
    return labels[..., np.newaxis]         # (N,H,W,1)

def normalize_sample(x_sample):
    """x_sample shape (1,H,W,C)"""
    x = x_sample.astype(np.float32)
    maxv = np.max(x[..., 0])
    if maxv > 0:
        x[..., 0] = x[..., 0] / maxv
    return x

def build_and_load_model(model_weights,
                         backbone="vgg16",
                         input_shape=(256,256,3),
                         classes=4,
                         lr=1e-4,
                         in_channels=None):
    """构建 model 并加载权重（返回已编译的 model）"""
    sm.set_framework('tf.keras')
    base_model = sm.Unet(backbone_name=backbone, input_shape=input_shape,
                         classes=classes, activation="softmax",
                         encoder_weights="imagenet")
    # 如果原始 x 有不同通道数，用 1x1 conv 映射到 3
    if in_channels is None:
        in_channels = 3
    inp = Input(shape=(None, None, in_channels))
    l1 = Conv2D(3, (1,1))(inp)
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)

    optim = Adam(learning_rate=lr)
    dice = sm.losses.DiceLoss(class_weights=np.array([1,1,1,0.5]))
    focal = sm.losses.CategoricalFocalLoss()
    total_loss = dice + (1 * focal)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim, total_loss, metrics)

    if not Path(model_weights).exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights}")
    logger.info(f"Loading weights from {model_weights}")
    model.load_weights(str(model_weights))
    return model

def save_fast_result(data_path, save_dir,
                            model_weights="modelUnet_10epochs.keras",
                            backbone="vgg16",
                            input_shape=(256,256,3),
                            classes=4,
                            k=5,
                            predict_batch_size=1,
                            lr=1e-4):
    """
    内存安全版：使用 mmap 逐样本读取并预测、绘图、保存。
    - data_path: 包含 x_2d_*.npy / y_2d_*.npy 的目录
    - save_dir: 保存图像目录（不存在则创建）
    - model_weights: 权重文件路径
    - k: 每个集合绘制样本数（取前 k，若某集合不足则使用最小数量）
    返回 saved_paths 列表
    """
    data_path = Path(data_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def npy_path(name): return data_path / name
    required = ["x_2d_train.npy", "y_2d_train.npy",
                "x_2d_val.npy", "y_2d_val.npy",
                "x_2d_test.npy", "y_2d_test.npy"]
    for r in required:
        if not npy_path(r).exists():
            raise FileNotFoundError(f"Missing required file: {npy_path(r)}")

    logger.info(f"Opening npy files with mmap_mode='r' (no eager full load).")
    x_train = np.load(str(npy_path("x_2d_train.npy")), mmap_mode='r')
    y_train = np.load(str(npy_path("y_2d_train.npy")), mmap_mode='r')
    x_val   = np.load(str(npy_path("x_2d_val.npy")), mmap_mode='r')
    y_val   = np.load(str(npy_path("y_2d_val.npy")), mmap_mode='r')
    x_test  = np.load(str(npy_path("x_2d_test.npy")), mmap_mode='r')
    y_test  = np.load(str(npy_path("y_2d_test.npy")), mmap_mode='r')

    logger.info(f"Shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # 确定 k 的实际可用值（不能超过任一数据集长度）
    k_eff = min(k, x_train.shape[0], x_val.shape[0], x_test.shape[0])
    if k_eff < k:
        logger.warning("Requested k=%d but reduced to %d due to dataset sizes", k, k_eff)
    k = k_eff

    # 构建并加载模型（一次）
    model = build_and_load_model(model_weights,
                                 backbone=backbone,
                                 input_shape=input_shape,
                                 classes=classes,
                                 lr=lr,
                                 in_channels=x_train.shape[-1])

    saved_paths = []
    try:
        for i in range(k):
            logger.info(f"Processing sample {i+1}/{k}")

            # --- 逐样本读取（mmap 切片不会复制全部数组） ---
            x_tr_s = normalize_sample(x_train[i:i+1])
            x_val_s = normalize_sample(x_val[i:i+1])
            x_test_s = normalize_sample(x_test[i:i+1])

            # 预测（小 batch）
            y_pred_train = model.predict(x_tr_s, batch_size=predict_batch_size, verbose=0)
            y_pred_val   = model.predict(x_val_s, batch_size=predict_batch_size, verbose=0)
            y_pred_test  = model.predict(x_test_s, batch_size=predict_batch_size, verbose=0)

            # 转成标签 (N,H,W,1)
            y_pred_train_lbl = pred_probs_to_label(y_pred_train)
            y_pred_val_lbl   = pred_probs_to_label(y_pred_val)
            y_pred_test_lbl  = pred_probs_to_label(y_pred_test)

            # 真实标签直接从 mmap 读取（通常为单通道整数），确保形状 (1,H,W,1)
            y_true_train_lbl = np.array(y_train[i:i+1])  # copy 1 sample 到内存（小）
            y_true_val_lbl   = np.array(y_val[i:i+1])
            y_true_test_lbl  = np.array(y_test[i:i+1])

            # 绘图并保存
            H, W = x_tr_s.shape[1], x_tr_s.shape[2]
            fig = plt.figure(figsize=(12,12))

            # Train row
            ax = plt.subplot(3,3,1)
            ax.imshow(x_tr_s[0,...,0], cmap='gray', interpolation='none'); ax.axis('off'); ax.set_title(f"Train Image {i}")
            ax = plt.subplot(3,3,2)
            ax.imshow(x_tr_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_true_train_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("True Mask")
            ax = plt.subplot(3,3,3)
            ax.imshow(x_tr_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_pred_train_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("Pred Mask")

            # Val row
            ax = plt.subplot(3,3,4)
            ax.imshow(x_val_s[0,...,0], cmap='gray', interpolation='none'); ax.axis('off'); ax.set_title(f"Val Image {i}")
            ax = plt.subplot(3,3,5)
            ax.imshow(x_val_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_true_val_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("True Mask")
            ax = plt.subplot(3,3,6)
            ax.imshow(x_val_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_pred_val_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("Pred Mask")

            # Test row
            ax = plt.subplot(3,3,7)
            ax.imshow(x_test_s[0,...,0], cmap='gray', interpolation='none'); ax.axis('off'); ax.set_title(f"Test Image {i}")
            ax = plt.subplot(3,3,8)
            ax.imshow(x_test_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_true_test_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("True Mask")
            ax = plt.subplot(3,3,9)
            ax.imshow(x_test_s[0,...,0], cmap='gray', interpolation='none')
            ax.imshow(y_pred_test_lbl[0,...,0], cmap='jet', interpolation='none', alpha=0.7); ax.axis('off'); ax.set_title("Pred Mask")

            plt.tight_layout()
            plot_path = save_dir / f"plot_{i}.png"
            plt.savefig(str(plot_path), dpi=100, bbox_inches='tight')
            plt.close(fig)
            saved_paths.append(plot_path)

            # 删除本次循环产生的大片内存，立即回收
            del x_tr_s, x_val_s, x_test_s
            del y_pred_train, y_pred_val, y_pred_test
            del y_pred_train_lbl, y_pred_val_lbl, y_pred_test_lbl
            del y_true_train_lbl, y_true_val_lbl, y_true_test_lbl
            gc.collect()

    finally:
        # 退出时确保释放 TF session
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        del model
        gc.collect()

    logger.info(f"Saved plots to {str(save_dir)}")
    return saved_paths
