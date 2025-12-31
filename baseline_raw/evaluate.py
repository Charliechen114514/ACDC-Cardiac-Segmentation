import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from scipy.spatial.distance import directed_hausdorff
import json
import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam

# ---------------------------
# GPU 设置
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---------------------------
# 参数
# ---------------------------
path = ""
RESULTS_DIR = "./evaluation_results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

NUM_CLASSES = 4  # RV, Myocardium, LV, Background
CLASS_NAMES = ["RV", "Myocardium", "LV", "Background"]

# ---------------------------
# 工具函数
# ---------------------------
def normalize(x):
    for i in range(x.shape[0]):
        max_val = np.max(x[i, :, :, 0])
        if max_val > 0:
            x[i, :, :, 0] /= max_val
    return x

def create_onehot(y_data):
    dim1, dim2, dim3, _ = y_data.shape
    y_new = np.zeros((dim1, dim2, dim3, 4), dtype=np.float32)
    for i in range(dim1):
        y_new[i, :, :, 0] = (y_data[i, :, :, 0] == 1)
        y_new[i, :, :, 1] = (y_data[i, :, :, 0] == 2)
        y_new[i, :, :, 2] = (y_data[i, :, :, 0] == 3)
        y_new[i, :, :, 3] = (y_data[i, :, :, 0] == 0)
    return y_new

def back_to_1_channel_mask(img, alpha=0.5):
    yy = np.zeros((img.shape[0], 256, 256))
    yy += 1.0 * (img[:, :, :, 0] >= alpha)
    yy += 2.0 * (img[:, :, :, 1] >= alpha)
    yy += 3.0 * (img[:, :, :, 2] >= alpha)
    return yy

def calculate_dice(y_true, y_pred):
    dice_scores = []
    for i in range(1, 4):
        gt = (y_true == i).astype(np.uint8)
        pred = (y_pred == i).astype(np.uint8)
        intersection = np.sum(gt * pred)
        sum_val = np.sum(gt) + np.sum(pred)
        dice_scores.append((2. * intersection / sum_val) if sum_val > 0 else 1.0)
    return dice_scores

def calculate_hausdorff(y_true, y_pred):
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
    predictions = []
    n_samples = x_data.shape[0]
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_pred = model.predict(x_data[i:end_idx], verbose=0)
        predictions.append(batch_pred)
        tf.keras.backend.clear_session()
        gc.collect()
    return np.concatenate(predictions, axis=0)

# ---------------------------
# 加载数据
# ---------------------------
print("Loading data...")
x_train = normalize(np.load(path + "x_2d_train.npy"))
y_train = np.load(path + "y_2d_train.npy")
x_val = normalize(np.load(path + "x_2d_val.npy"))
y_val = np.load(path + "y_2d_val.npy")
x_test = normalize(np.load(path + "x_2d_test.npy"))
y_test = np.load(path + "y_2d_test.npy")

y_train_oh = create_onehot(y_train)
y_val_oh = create_onehot(y_val)
y_test_oh = create_onehot(y_test)

# ---------------------------
# 构建模型
# ---------------------------
print("Building model...")
base_model = sm.Unet(backbone_name="vgg16", input_shape=(256, 256, 3), classes=4, activation="softmax", encoder_weights="imagenet")
inp = Input(shape=(None, None, x_train.shape[-1]))
l1 = Conv2D(3, (1, 1))(inp)
out = base_model(l1)
modelUnet = Model(inp, out, name=base_model.name)
modelUnet.compile(Adam(0.0001), sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(),
                  metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)])

print("Loading weights...")
modelUnet.load_weights("risultati100epochs/modelUnet_50epochs.keras")

# ---------------------------
# 预测与评估
# ---------------------------
num_eval = min(len(x_test), 50)
print(f"Predicting {num_eval} test samples...")
y_pred_test = batch_predict(modelUnet, x_test[:num_eval], batch_size=2)

all_dice, all_hd, all_metrics = [], [], []
for i in range(num_eval):
    true_mask = back_to_1_channel_mask(y_test_oh[i:i+1])[0]
    pred_mask = back_to_1_channel_mask(y_pred_test[i:i+1])[0]
    
    all_dice.append(calculate_dice(true_mask, pred_mask))
    all_hd.append(calculate_hausdorff(true_mask, pred_mask))
    all_metrics.append(calculate_additional_metrics(true_mask, pred_mask))

all_dice = np.array(all_dice)
all_hd = np.array(all_hd)
all_metrics = np.array(all_metrics)  # shape: (num_eval, 3, 5)

# ---------------------------
# 保存 CSV
# ---------------------------
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
print(f"Saved detailed CSV metrics to {csv_path}")

# ---------------------------
# 保存 JSON summary
# ---------------------------
# ---------------------------
# 保存 JSON summary（按类别拆开）
# ---------------------------
summary_per_class = {
    "num_test_samples": num_eval,
    "dice_mean": {CLASS_NAMES[i]: float(all_dice[:, i].mean()) for i in range(3)},
    "hd_mean": {CLASS_NAMES[i]: float(all_hd[:, i].mean()) for i in range(3)},
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

json_path_per_class = os.path.join(RESULTS_DIR, "test_metrics_summary_per_class.json")
with open(json_path_per_class, 'w') as f:
    json.dump(summary_per_class, f, indent=4)

print(f"Saved per-class summary JSON to {json_path_per_class}")


# ---------------------------
# 可视化前 5 个样本
# ---------------------------
os.makedirs(PLOT_DIR, exist_ok=True)
for i in range(min(num_eval, 10)):
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
    plt.savefig(os.path.join(PLOT_DIR, f"eval_{i}.png"))
    plt.close()

print("Evaluation completed.")

