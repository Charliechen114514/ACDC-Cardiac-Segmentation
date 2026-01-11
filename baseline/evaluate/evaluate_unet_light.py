import os
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from scipy.spatial.distance import directed_hausdorff
import json
import segmentation_models as sm
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    concatenate, BatchNormalization, Activation, Dropout
)
from keras.models import Model
from keras.optimizers import Adam
from loguru import logger

# ---------------------------
# GPU 设置
# ---------------------------
def setup_gpu():
    """配置 GPU 内存增长"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"Failed to set GPU memory growth: {e}")
    else:
        logger.info("No GPU found, using CPU")

# ---------------------------
# 轻量级 U-Net 模型构建（与训练脚本完全一致）
# ---------------------------
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
    """
    inputs = Input(shape=input_shape)
    
    # ============ 编码器（Encoder / 下采样路径）============
    conv1 = conv_block(inputs, 32, dropout_rate=0.1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 64, dropout_rate=0.1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 128, dropout_rate=0.2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 256, dropout_rate=0.2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # ============ 瓶颈层（Bottleneck）============
    conv5 = conv_block(pool4, 512, dropout_rate=0.3)
    
    # ============ 解码器（Decoder / 上采样路径）============
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(256, (2, 2), padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(merge6, 256, dropout_rate=0.2)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 128, dropout_rate=0.2)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(64, (2, 2), padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 64, dropout_rate=0.1)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 32, dropout_rate=0.1)
    
    # ============ 输出层 ============
    outputs = Conv2D(num_classes, (1, 1), activation=activation, name='output')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs, name='Lightweight_UNet')
    
    return model

# ---------------------------
# 工具函数
# ---------------------------
def normalize(x):
    """归一化输入数据"""
    for i in range(x.shape[0]):
        max_val = np.max(x[i, :, :, 0])
        if max_val > 0:
            x[i, :, :, 0] /= max_val
    return x

def create_onehot(y_data):
    """创建 one-hot 编码"""
    dim1, dim2, dim3, _ = y_data.shape
    y_new = np.zeros((dim1, dim2, dim3, 4), dtype=np.float32)
    for i in range(dim1):
        y_new[i, :, :, 0] = (y_data[i, :, :, 0] == 1)
        y_new[i, :, :, 1] = (y_data[i, :, :, 0] == 2)
        y_new[i, :, :, 2] = (y_data[i, :, :, 0] == 3)
        y_new[i, :, :, 3] = (y_data[i, :, :, 0] == 0)
    return y_new

def back_to_1_channel_mask(img, alpha=0.5):
    """将多通道预测转换回单通道 mask"""
    yy = np.zeros((img.shape[0], 256, 256))
    yy += 1.0 * (img[:, :, :, 0] >= alpha)
    yy += 2.0 * (img[:, :, :, 1] >= alpha)
    yy += 3.0 * (img[:, :, :, 2] >= alpha)
    return yy

def calculate_dice(y_true, y_pred):
    """计算 Dice 系数"""
    dice_scores = []
    for i in range(1, 4):
        gt = (y_true == i).astype(np.uint8)
        pred = (y_pred == i).astype(np.uint8)
        intersection = np.sum(gt * pred)
        sum_val = np.sum(gt) + np.sum(pred)
        dice_scores.append((2. * intersection / sum_val) if sum_val > 0 else 1.0)
    return dice_scores

def calculate_hausdorff(y_true, y_pred):
    """计算 Hausdorff 距离"""
    hd_scores = []
    for i in range(1, 4):
        gt_points = np.argwhere(y_true == i)
        pred_points = np.argwhere(y_pred == i)
        if len(gt_points) == 0 and len(pred_points) == 0:
            hd_scores.append(0.0)
        elif len(gt_points) == 0 or len(pred_points) == 0:
            hd_scores.append(100.0)
        else:
            d1 = directed_hausdorff(gt_points, pred_points)[0]
            d2 = directed_hausdorff(pred_points, gt_points)[0]
            hd_scores.append(max(d1, d2))
    return hd_scores

def calculate_additional_metrics(y_true, y_pred):
    """计算额外的评估指标"""
    metrics = []
    for i in range(1, 4):
        gt = (y_true == i).astype(np.uint8)
        pred = (y_pred == i).astype(np.uint8)
        TP = np.sum(gt * pred)
        FP = np.sum((1 - gt) * pred)
        FN = np.sum(gt * (1 - pred))
        TN = np.sum((1 - gt) * (1 - pred))
        acc = (TP + TN) / (TP + FP + FN + TN + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        specificity = TN / (TN + FP + 1e-7)
        f1 = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        iou = TP / (TP + FP + FN + 1e-7)
        metrics.append([acc, recall, specificity, f1, iou])
    return metrics

def batch_predict(model, x_data, batch_size=4):
    """批量预测以避免内存溢出"""
    predictions = []
    n_samples = x_data.shape[0]
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_pred = model.predict(x_data[i:end_idx], verbose=0)
        predictions.append(batch_pred)
        tf.keras.backend.clear_session()
        gc.collect()
    return np.concatenate(predictions, axis=0)

def get_eval_string_from_path(file_path):
    """
    从模型路径中提取 epoch 数字并生成 eval 字符串
    """
    file_name = os.path.basename(file_path)
    match = re.search(r'(\d+)', file_name)
    
    if match:
        epoch = match.group(1)
        return f"lightweight_unet_{epoch}_model_eval"
    else:
        return "lightweight_unet_eval"

# ---------------------------
# 主评估函数
# ---------------------------
def evaluate_unetlight(EVA_BASE_FOLDER, data_folder, model_weights_path, num_visualize=10):
    """
    评估训练好的轻量级 U-Net 分割模型
    
    参数:
        EVA_BASE_FOLDER: 评估结果保存文件夹路径
        data_folder: 数据文件夹路径（包含 .npy 文件）
        model_weights_path: 模型权重文件路径
        num_visualize: 生成可视化图像的数量，默认 10
    """
    # ---------------------------
    # 初始化
    # ---------------------------
    setup_gpu()
    
    logger.info("=" * 60)
    logger.info("Starting Lightweight U-Net Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Model weights path: {model_weights_path}")
    logger.info(f"Number of visualizations to generate: {num_visualize}")
    
    path = EVA_BASE_FOLDER
    RESULTS_DIR = os.path.join(path, get_eval_string_from_path(model_weights_path))
    PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    logger.info(f"Results directory created: {RESULTS_DIR}")
    logger.info(f"Plots directory created: {PLOT_DIR}")
    
    NUM_CLASSES = 4  # RV, Myocardium, LV, Background
    CLASS_NAMES = ["RV", "Myocardium", "LV", "Background"]
    logger.info(f"Number of classes: {NUM_CLASSES}")
    logger.info(f"Class names: {CLASS_NAMES}")
    
    # ---------------------------
    # 加载数据
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Loading data...")
    
    x_train_path = os.path.join(data_folder, "x_2d_train.npy")
    y_train_path = os.path.join(data_folder, "y_2d_train.npy")
    x_val_path = os.path.join(data_folder, "x_2d_val.npy")
    y_val_path = os.path.join(data_folder, "y_2d_val.npy")
    x_test_path = os.path.join(data_folder, "x_2d_test.npy")
    y_test_path = os.path.join(data_folder, "y_2d_test.npy")
    
    logger.info(f"Loading training data from {x_train_path}")
    x_train = normalize(np.load(x_train_path))
    y_train = np.load(y_train_path)
    logger.info(f"✓ Training data loaded: x_train shape {x_train.shape}, y_train shape {y_train.shape}")
    
    logger.info(f"Loading validation data from {x_val_path}")
    x_val = normalize(np.load(x_val_path))
    y_val = np.load(y_val_path)
    logger.info(f"✓ Validation data loaded: x_val shape {x_val.shape}, y_val shape {y_val.shape}")
    
    logger.info(f"Loading test data from {x_test_path}")
    x_test = normalize(np.load(x_test_path))
    y_test = np.load(y_test_path)
    logger.info(f"✓ Test data loaded: x_test shape {x_test.shape}, y_test shape {y_test.shape}")
    
    logger.info("Creating one-hot encodings...")
    y_train_oh = create_onehot(y_train)
    y_val_oh = create_onehot(y_val)
    y_test_oh = create_onehot(y_test)
    logger.info("✓ One-hot encodings created")
    
    # ---------------------------
    # 构建轻量级 U-Net 模型
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Building Lightweight U-Net model...")
    logger.info("Model type: Lightweight U-Net (No Pretrained Backbone)")
    logger.info("Input shape: (256, 256, 1)")
    logger.info("Classes: 4")
    logger.info("Activation: softmax")
    
    modelUnet = build_lightweight_unet(
        input_shape=(256, 256, 1),
        num_classes=NUM_CLASSES,
        activation='softmax'
    )
    
    # 编译模型（使用与训练时相同的配置）
    modelUnet.compile(
        Adam(0.0001), 
        sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5])) + sm.losses.CategoricalFocalLoss(),
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    )
    
    logger.info("✓ Model built and compiled successfully")
    
    # 打印模型参数量
    total_params = modelUnet.count_params()
    logger.info(f"Total model parameters: {total_params:,}")
    
    # ---------------------------
    # 加载模型权重
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Loading model weights...")
    if not os.path.exists(model_weights_path):
        logger.error(f"No model found in {model_weights_path}")
        return 
    
    modelUnet.load_weights(model_weights_path)
    logger.info(f"✓ Model weights loaded from {model_weights_path}")
    
    # ---------------------------
    # 预测与评估
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Starting prediction and evaluation...")
    
    num_eval = min(len(x_test), 50)
    logger.info(f"Number of test samples to evaluate: {num_eval}")
    logger.info(f"Predicting {num_eval} test samples with batch size 2...")
    
    y_pred_test = batch_predict(modelUnet, x_test[:num_eval], batch_size=2)
    logger.info(f"✓ Predictions completed, shape: {y_pred_test.shape}")
    
    logger.info("Calculating metrics for each sample...")
    all_dice, all_hd, all_metrics = [], [], []
    for i in range(num_eval):
        true_mask = back_to_1_channel_mask(y_test_oh[i:i+1])[0]
        pred_mask = back_to_1_channel_mask(y_pred_test[i:i+1])[0]
        
        all_dice.append(calculate_dice(true_mask, pred_mask))
        all_hd.append(calculate_hausdorff(true_mask, pred_mask))
        all_metrics.append(calculate_additional_metrics(true_mask, pred_mask))
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{num_eval} samples")
    
    all_dice = np.array(all_dice)
    all_hd = np.array(all_hd)
    all_metrics = np.array(all_metrics)  # shape: (num_eval, 3, 5)
    logger.info(f"✓ Metrics calculated for all {num_eval} samples")
    logger.info(f"  Dice scores shape: {all_dice.shape}")
    logger.info(f"  Hausdorff distances shape: {all_hd.shape}")
    logger.info(f"  Additional metrics shape: {all_metrics.shape}")
    
    # ---------------------------
    # 保存 CSV
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Saving detailed metrics to CSV...")
    
    rows = []
    for i in range(num_eval):
        row = {
            "sample_id": i,
            "dice_rv": all_dice[i,0], "dice_myo": all_dice[i,1], "dice_lv": all_dice[i,2],
            "hd_rv": all_hd[i,0], "hd_myo": all_hd[i,1], "hd_lv": all_hd[i,2]
        }
        for c, name in enumerate(CLASS_NAMES[:-1]):
            row.update({
                f"{name}_acc": all_metrics[i, c, 0],
                f"{name}_recall": all_metrics[i, c, 1],
                f"{name}_specificity": all_metrics[i, c, 2],
                f"{name}_f1": all_metrics[i, c, 3],
                f"{name}_iou": all_metrics[i, c, 4],
            })
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "test_metrics_detailed.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Detailed CSV metrics saved to {csv_path}")
    logger.info(f"  CSV contains {len(df)} rows and {len(df.columns)} columns")
    
    # ---------------------------
    # 保存 JSON summary（按类别拆开）
    # ---------------------------
    logger.info("-" * 60)
    logger.info("Generating summary statistics...")
    
    summary_per_class = {
        "model_type": "Lightweight U-Net (No Pretrained Backbone)",
        "total_parameters": int(total_params),
        "num_test_samples": num_eval,
        "dice_mean": {CLASS_NAMES[i]: float(all_dice[:, i].mean()) for i in range(3)},
        "dice_std": {CLASS_NAMES[i]: float(all_dice[:, i].std()) for i in range(3)},
        "hd_mean": {CLASS_NAMES[i]: float(all_hd[:, i].mean()) for i in range(3)},
        "hd_std": {CLASS_NAMES[i]: float(all_hd[:, i].std()) for i in range(3)},
        "metrics_mean": {},
        "metrics_std": {}
    }
    
    metric_names = ["accuracy", "recall", "specificity", "f1_score", "iou"]
    
    for c in range(3):
        summary_per_class["metrics_mean"][CLASS_NAMES[c]] = {
            metric_names[m]: float(all_metrics[:, c, m].mean()) for m in range(5)
        }
        summary_per_class["metrics_std"][CLASS_NAMES[c]] = {
            metric_names[m]: float(all_metrics[:, c, m].std()) for m in range(5)
        }
    
    # 打印汇总统计信息
    logger.info("Summary Statistics:")
    logger.info(f"  Model: Lightweight U-Net")
    logger.info(f"  Total Parameters: {total_params:,}")
    for class_name in CLASS_NAMES[:-1]:
        logger.info(f"  {class_name}:")
        logger.info(f"    Mean Dice: {summary_per_class['dice_mean'][class_name]:.4f} ± {summary_per_class['dice_std'][class_name]:.4f}")
        logger.info(f"    Mean Hausdorff Distance: {summary_per_class['hd_mean'][class_name]:.4f} ± {summary_per_class['hd_std'][class_name]:.4f}")
        logger.info(f"    Mean F1 Score: {summary_per_class['metrics_mean'][class_name]['f1_score']:.4f}")
        logger.info(f"    Mean IoU: {summary_per_class['metrics_mean'][class_name]['iou']:.4f}")
    
    json_path_per_class = os.path.join(RESULTS_DIR, "test_metrics_summary_per_class.json")
    with open(json_path_per_class, 'w') as f:
        json.dump(summary_per_class, f, indent=4)
    
    logger.info(f"✓ Per-class summary JSON saved to {json_path_per_class}")
    
    # ---------------------------
    # 可视化样本
    # ---------------------------
    logger.info("-" * 60)
    logger.info(f"Generating visualizations for {min(num_eval, num_visualize)} samples...")
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    for i in range(min(num_eval, num_visualize)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        true_mask = back_to_1_channel_mask(y_test_oh[i:i+1])[0]
        pred_mask = back_to_1_channel_mask(y_pred_test[i:i+1])[0]
        
        axes[0].imshow(x_test[i, :, :, 0], cmap='gray')
        axes[0].set_title(f"Test Image {i}")
        
        axes[1].imshow(x_test[i, :, :, 0], cmap='gray')
        axes[1].imshow(true_mask, cmap='jet', alpha=0.5)
        axes[1].set_title("True Mask")
        
        axes[2].imshow(x_test[i, :, :, 0], cmap='gray')
        axes[2].imshow(pred_mask, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Pred (Dice: {np.mean(all_dice[i]):.2f})")
        
        for ax in axes: ax.axis('off')
        save_path = os.path.join(PLOT_DIR, f"eval_{i}.png")
        plt.savefig(save_path)
        plt.close()
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Generated {i + 1}/{min(num_eval, num_visualize)} visualizations")
    
    logger.info(f"✓ All visualizations saved to {PLOT_DIR}")
    
    # ---------------------------
    # 完成
    # ---------------------------
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved in: {RESULTS_DIR}")
    logger.info(f"  - Detailed CSV: {csv_path}")
    logger.info(f"  - Summary JSON: {json_path_per_class}")
    logger.info(f"  - Visualizations: {PLOT_DIR} ({min(num_eval, num_visualize)} images)")
    logger.info("=" * 60)