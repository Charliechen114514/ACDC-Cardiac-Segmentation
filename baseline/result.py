import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gc

import segmentation_models as sm

from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score

from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam

# ---------------------------
# GPU 内存设置
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

path = ""

# ---------------------------
# 工具函数：分批预测（减少内存占用）
# ---------------------------
def batch_predict(model, x_data, batch_size=4):
    """分批预测，降低内存占用"""
    predictions = []
    n_samples = x_data.shape[0]

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_pred = model.predict(x_data[i:end_idx], verbose=0)
        predictions.append(batch_pred)
        # 清理内存
        tf.keras.backend.clear_session()
        gc.collect()

    return np.concatenate(predictions, axis=0)

# ---------------------------
# 加载数据
# ---------------------------
print("Loading data...")
x_train = np.load(path + "x_2d_train.npy")
y_train = np.load(path + "y_2d_train.npy")

x_val = np.load(path + "x_2d_val.npy")
y_val = np.load(path + "y_2d_val.npy")

x_test = np.load(path + "x_2d_test.npy")
y_test = np.load(path + "y_2d_test.npy")

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ---------------------------
# 数据标准化
# ---------------------------
print("Normalizing data...")
def normalize(x):
    for i in range(x.shape[0]):
        max_val = np.max(x[i, :, :, 0])
        if max_val > 0:
            x[i, :, :, 0] /= max_val
    return x

x_train = normalize(x_train)
x_val = normalize(x_val)
x_test = normalize(x_test)

# ---------------------------
# one-hot 编码
# ---------------------------
print("Creating one-hot encoded masks...")
def create_onehot(y_data):
    dim1, dim2, dim3, _ = y_data.shape
    y_new = np.zeros((dim1, dim2, dim3, 4), dtype=np.float32)
    for i in range(dim1):
        y_new[i, :, :, 0] = (y_data[i, :, :, 0] == 1)
        y_new[i, :, :, 1] = (y_data[i, :, :, 0] == 2)
        y_new[i, :, :, 2] = (y_data[i, :, :, 0] == 3)
        y_new[i, :, :, 3] = (y_data[i, :, :, 0] == 0)
    return y_new


y_train = create_onehot(y_train)
y_val = create_onehot(y_val)
y_test = create_onehot(y_test)

# ---------------------------
# 构建模型
# ---------------------------
print("Building model...")
bb = "vgg16"
input_shape = (256, 256, 3)
c = 4
enc_weights = "imagenet"
activation = "softmax"
base_model = sm.Unet(backbone_name=bb, input_shape=input_shape, classes=c, activation=activation, encoder_weights=enc_weights)

N = x_train.shape[-1]
inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp)
out = base_model(l1)
modelUnet = Model(inp, out, name=base_model.name)

lr = 0.0001
optim = Adam(lr)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

modelUnet.compile(optim, total_loss, metrics)

print("Loading weights...")
modelUnet.load_weights("risultati100epochs/modelUnet_100epochs.keras")

os.makedirs("plot", exist_ok=True)

# ---------------------------
# 使用 batch_predict 避免 OOM
# ---------------------------
print("Predicting on all datasets...")
k = 5
batch_size = 2  # 可调节，避免 OOM

y_pred_train = batch_predict(modelUnet, x_train[:k], batch_size)
y_pred_val = batch_predict(modelUnet, x_val[:k], batch_size)
y_pred_test = batch_predict(modelUnet, x_test[:k], batch_size)

tf.keras.backend.clear_session()
gc.collect()

# ---------------------------
# 工具函数：多通道 mask -> 单通道
# ---------------------------
def back_to_1_channel_mask(img, alpha=0.5):
    yy = np.zeros((img.shape[0], 256, 256, 1))
    yy += 1.0 * (img[:, :, :, 0:1] >= alpha)
    yy += 2.0 * (img[:, :, :, 1:2] >= alpha)
    yy += 3.0 * (img[:, :, :, 2:3] >= alpha)
    return yy

# ---------------------------
# 绘图函数
# ---------------------------
def plot_single_comparison(n, x_data, y_true, y_pred, dataset_name):
    alpha = 0.5
    y_pred_1ch = back_to_1_channel_mask(y_pred[n:n+1], alpha)
    y_true_1ch = back_to_1_channel_mask(y_true[n:n+1], alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_data[n, :, :, 0], cmap='gray', interpolation='none')
    axes[0].set_title(f'{dataset_name} Image {n}')
    axes[0].axis('off')

    axes[1].imshow(x_data[n, :, :, 0], cmap='gray', interpolation='none')
    axes[1].imshow(y_true_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    axes[2].imshow(x_data[n, :, :, 0], cmap='gray', interpolation='none')
    axes[2].imshow(y_pred_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    return fig

# ---------------------------
# 生成对比图
# ---------------------------
print("Generating plots...")
for i in range(k):
    print(f"Plotting {i+1} out of {k}")
    fig = plt.figure(figsize=(12, 12))

    # Train
    plt.subplot(3, 3, 1)
    plt.imshow(x_train[i, :, :, 0], cmap='gray', interpolation='none')
    plt.title(f"Train Image {i}")
    plt.axis('off')

    plt.subplot(3, 3, 2)
    y_true_train = back_to_1_channel_mask(y_train[i:i+1], 0.5)
    plt.imshow(x_train[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_true_train[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    y_pred_train_1ch = back_to_1_channel_mask(y_pred_train[i:i+1], 0.5)
    plt.imshow(x_train[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_pred_train_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("Pred Mask")
    plt.axis('off')

    # Val
    plt.subplot(3, 3, 4)
    plt.imshow(x_val[i, :, :, 0], cmap='gray', interpolation='none')
    plt.title(f"Val Image {i}")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    y_true_val = back_to_1_channel_mask(y_val[i:i+1], 0.5)
    plt.imshow(x_val[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_true_val[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    y_pred_val_1ch = back_to_1_channel_mask(y_pred_val[i:i+1], 0.5)
    plt.imshow(x_val[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_pred_val_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("Pred Mask")
    plt.axis('off')

    # Test
    plt.subplot(3, 3, 7)
    plt.imshow(x_test[i, :, :, 0], cmap='gray', interpolation='none')
    plt.title(f"Test Image {i}")
    plt.axis('off')

    plt.subplot(3, 3, 8)
    y_true_test = back_to_1_channel_mask(y_test[i:i+1], 0.5)
    plt.imshow(x_test[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_true_test[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(3, 3, 9)
    y_pred_test_1ch = back_to_1_channel_mask(y_pred_test[i:i+1], 0.5)
    plt.imshow(x_test[i, :, :, 0], cmap='gray', interpolation='none')
    plt.imshow(y_pred_test_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
    plt.title("Pred Mask")
    plt.axis('off')

    plt.tight_layout()
    plot_name = f"plot/plot_{i}.png"
    plt.savefig(plot_name, dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()

print("All plots generated successfully!")
