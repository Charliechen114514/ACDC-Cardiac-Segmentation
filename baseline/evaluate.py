# # import os
# # import numpy as np
# # import tensorflow as tf
# # import matplotlib.pyplot as plt
# # from scipy.ndimage import distance_transform_edt, binary_erosion
# # import pandas as pd
# # from tqdm import tqdm
# # import json
# # import gc

# # # -------------------------------
# # # GPU 设置
# # # -------------------------------
# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # if gpus:
# #     try:
# #         for gpu in gpus:
# #             tf.config.experimental.set_memory_growth(gpu, True)
# #     except RuntimeError as e:
# #         print(e)

# # # -------------------------------
# # # 参数设置
# # # -------------------------------
# # NUM_CLASSES = 4
# # CLASS_NAMES = ["RV", "Myocardium", "LV", "Background"]
# # MODEL_PATH = "./risultati100epochs/modelUnet_100epochs.keras"
# # RESULTS_DIR = "./evaluation_results"
# # PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
# # os.makedirs(RESULTS_DIR, exist_ok=True)
# # os.makedirs(PLOT_DIR, exist_ok=True)

# # # 数据路径
# # path = ""
# # X_TRAIN_PATH = path + "x_2d_train.npy"
# # Y_TRAIN_PATH = path + "y_2d_train.npy"
# # X_VAL_PATH = path + "x_2d_val.npy"
# # Y_VAL_PATH = path + "y_2d_val.npy"
# # X_TEST_PATH = path + "x_2d_test.npy"
# # Y_TEST_PATH = path + "y_2d_test.npy"

# # # 可视化参数
# # NUM_SAMPLES_TO_PLOT = 5
# # BATCH_SIZE_PREDICT = 2  # 分批预测，降低内存占用

# # # -------------------------------
# # # 工具函数
# # # -------------------------------
# # def normalize(x):
# #     """逐样本标准化"""
# #     x_normalized = x.copy()
# #     for i in range(x.shape[0]):
# #         max_val = np.max(x_normalized[i, :, :, 0])
# #         if max_val > 0:
# #             x_normalized[i, :, :, 0] /= max_val
# #     return x_normalized

# # def create_onehot(y_data):
# #     """创建one-hot编码的mask"""
# #     dim1, dim2, dim3, _ = y_data.shape
# #     y_new = np.zeros((dim1, dim2, dim3, 4), dtype=np.float32)
# #     for i in range(dim1):
# #         y_new[i, :, :, 0] = (y_data[i, :, :, 0] == 1)
# #         y_new[i, :, :, 1] = (y_data[i, :, :, 0] == 2)
# #         y_new[i, :, :, 2] = (y_data[i, :, :, 0] == 3)
# #         y_new[i, :, :, 3] = (y_data[i, :, :, 0] == 0)
# #     return y_new

# # def back_to_1_channel_mask(img, alpha=0.5):
# #     yy = np.zeros((img.shape[0], 256, 256, 1))
# #     yy += 1.0 * (img[:, :, :, 0:1] >= alpha)
# #     yy += 2.0 * (img[:, :, :, 1:2] >= alpha)
# #     yy += 3.0 * (img[:, :, :, 2:3] >= alpha)
# #     return yy

# # def dice_coefficient(y_true, y_pred, class_id, smooth=1e-7):
# #     y_true_class = y_true[:, :, class_id]
# #     y_pred_class = y_pred[:, :, class_id]
# #     intersection = np.sum(y_true_class * y_pred_class)
# #     union = np.sum(y_true_class) + np.sum(y_pred_class)
# #     dice = (2. * intersection + smooth) / (union + smooth)
# #     return dice

# # def hausdorff_distance_95(y_true, y_pred, class_id, spacing=(1.0, 1.0)):
# #     mask_true = y_true[:, :, class_id] > 0.5
# #     mask_pred = y_pred[:, :, class_id] > 0.5
# #     if not np.any(mask_true) or not np.any(mask_pred):
# #         return np.nan
# #     boundary_true = mask_true & ~binary_erosion(mask_true)
# #     boundary_pred = mask_pred & ~binary_erosion(mask_pred)
# #     if not np.any(boundary_true) or not np.any(boundary_pred):
# #         return np.nan
# #     dist_true = distance_transform_edt(~boundary_true, sampling=spacing)
# #     dist_pred = distance_transform_edt(~boundary_pred, sampling=spacing)
# #     distances_true_to_pred = dist_true[boundary_true]
# #     distances_pred_to_true = dist_pred[boundary_pred]
# #     all_distances = np.concatenate([distances_true_to_pred, distances_pred_to_true])
# #     hd95 = np.percentile(all_distances, 95)
# #     return hd95

# # def plot_comparison_grid(idx, x_train, y_train, y_pred_train,
# #                          x_val, y_val, y_pred_val,
# #                          x_test, y_test, y_pred_test, save_path):
# #     fig = plt.figure(figsize=(12, 12))
# #     alpha = 0.5
    
# #     # Train
# #     plt.subplot(3, 3, 1)
# #     plt.imshow(x_train[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.title(f"Train Image {idx}")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 2)
# #     y_true_train = back_to_1_channel_mask(y_train[idx:idx+1], alpha)
# #     plt.imshow(x_train[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_true_train[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("True Mask")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 3)
# #     y_pred_train_1ch = back_to_1_channel_mask(y_pred_train[idx:idx+1], alpha)
# #     plt.imshow(x_train[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_pred_train_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("Pred Mask")
# #     plt.axis('off')
    
# #     # Val
# #     plt.subplot(3, 3, 4)
# #     plt.imshow(x_val[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.title(f"Val Image {idx}")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 5)
# #     y_true_val = back_to_1_channel_mask(y_val[idx:idx+1], alpha)
# #     plt.imshow(x_val[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_true_val[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("True Mask")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 6)
# #     y_pred_val_1ch = back_to_1_channel_mask(y_pred_val[idx:idx+1], alpha)
# #     plt.imshow(x_val[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_pred_val_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("Pred Mask")
# #     plt.axis('off')
    
# #     # Test
# #     plt.subplot(3, 3, 7)
# #     plt.imshow(x_test[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.title(f"Test Image {idx}")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 8)
# #     y_true_test = back_to_1_channel_mask(y_test[idx:idx+1], alpha)
# #     plt.imshow(x_test[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_true_test[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("True Mask")
# #     plt.axis('off')
    
# #     plt.subplot(3, 3, 9)
# #     y_pred_test_1ch = back_to_1_channel_mask(y_pred_test[idx:idx+1], alpha)
# #     plt.imshow(x_test[idx, :, :, 0], cmap='gray', interpolation='none')
# #     plt.imshow(y_pred_test_1ch[0, :, :, 0], cmap='jet', interpolation='none', alpha=0.7)
# #     plt.title("Pred Mask")
# #     plt.axis('off')
    
# #     plt.tight_layout()
# #     plt.savefig(save_path, dpi=100, bbox_inches='tight')
# #     plt.close(fig)
# #     gc.collect()

# # # -------------------------------
# # # 主评估函数 (OOM-safe)
# # # -------------------------------
# # def evaluate_model():
# #     print("=" * 70)
# #     print("ACDC 模型完整评估 (OOM-safe)")
# #     print("=" * 70)

# #     # ---------------------------
# #     # 1. 加载模型
# #     # ---------------------------
# #     print(f"\n[1/6] 加载模型: {MODEL_PATH}")
# #     if not os.path.exists(MODEL_PATH):
# #         print(f"错误：模型文件不存在: {MODEL_PATH}")
# #         return
# #     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# #     print("✓ 模型加载成功！")

# #     # ---------------------------
# #     # 2. 流式处理数据集
# #     # ---------------------------
# #     def process_dataset(x_path, y_path, dataset_name):
# #         n_samples = np.load(x_path, mmap_mode='r').shape[0]
# #         dice_list, hd_list = [], []
# #         y_pred_all = []

# #         for i in tqdm(range(0, n_samples, BATCH_SIZE_PREDICT), desc=f"预测 {dataset_name}"):
# #             end_idx = min(i + BATCH_SIZE_PREDICT, n_samples)
# #             x_batch = np.load(x_path, mmap_mode='r')[i:end_idx]
# #             y_batch = np.load(y_path, mmap_mode='r')[i:end_idx]

# #             x_batch = normalize(x_batch)
# #             y_batch_onehot = create_onehot(y_batch)

# #             y_pred_batch = model.predict(x_batch, verbose=0)

# #             # 保存预测
# #             y_pred_all.append(y_pred_batch)

# #             # 指标计算（逐样本）
# #             for j in range(y_batch.shape[0]):
# #                 gt = y_batch_onehot[j]
# #                 pred = y_pred_batch[j]

# #                 dice_scores = [dice_coefficient(gt, pred, c) for c in range(NUM_CLASSES)]
# #                 hd_scores = [hausdorff_distance_95(gt, pred, c) for c in range(NUM_CLASSES)]

# #                 dice_list.append(dice_scores)
# #                 hd_list.append(hd_scores)

# #             del x_batch, y_batch, y_batch_onehot, y_pred_batch
# #             gc.collect()

# #         y_pred_all = np.concatenate(y_pred_all, axis=0)
# #         dice_arr = np.array(dice_list)
# #         hd_arr = np.array(hd_list)
# #         return dice_arr, hd_arr, y_pred_all

# #     print(f"\n[2/6] 预测数据集并计算指标...")
# #     dice_train, hd_train, y_pred_train = process_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, "Train")
# #     dice_val, hd_val, y_pred_val = process_dataset(X_VAL_PATH, Y_VAL_PATH, "Val")
# #     dice_test, hd_test, y_pred_test = process_dataset(X_TEST_PATH, Y_TEST_PATH, "Test")

# #     tf.keras.backend.clear_session()
# #     gc.collect()
# #     print("✓ 预测完成")

# #     # ---------------------------
# #     # 3. 生成 DataFrame
# #     # ---------------------------
# #     df = pd.DataFrame({
# #         'sample_id': np.arange(dice_test.shape[0]),
# #         'dice_rv': dice_test[:,0], 'dice_myo': dice_test[:,1],
# #         'dice_lv': dice_test[:,2], 'dice_bg': dice_test[:,3],
# #         'dice_mean': dice_test[:,:3].mean(axis=1),
# #         'hd95_rv': hd_test[:,0], 'hd95_myo': hd_test[:,1],
# #         'hd95_lv': hd_test[:,2], 'hd95_bg': hd_test[:,3],
# #         'hd95_mean': np.nanmean(hd_test[:,:3], axis=1)
# #     })

# #     # ---------------------------
# #     # 4. 可视化前 NUM_SAMPLES_TO_PLOT 个样本
# #     # ---------------------------
# #     print(f"\n[3/6] 生成可视化对比图...")
# #     n_plot = min(NUM_SAMPLES_TO_PLOT, dice_test.shape[0])
# #     for i in tqdm(range(n_plot), desc="生成对比图"):
# #         plot_path = os.path.join(PLOT_DIR, f"comparison_{i}.png")
# #         plot_comparison_grid(
# #             i,
# #             np.load(X_TRAIN_PATH, mmap_mode='r'),
# #             create_onehot(np.load(Y_TRAIN_PATH, mmap_mode='r')),
# #             y_pred_train,
# #             np.load(X_VAL_PATH, mmap_mode='r'),
# #             create_onehot(np.load(Y_VAL_PATH, mmap_mode='r')),
# #             y_pred_val,
# #             np.load(X_TEST_PATH, mmap_mode='r'),
# #             create_onehot(np.load(Y_TEST_PATH, mmap_mode='r')),
# #             y_pred_test,
# #             plot_path
# #         )
# #     print(f"✓ 可视化完成，保存至: {PLOT_DIR}/")

# #     # ---------------------------
# #     # 5. 保存 CSV/JSON
# #     # ---------------------------
# #     csv_path = os.path.join(RESULTS_DIR, "test_metrics_detailed.csv")
# #     df.to_csv(csv_path, index=False)

# #     summary = {
# #         "model": MODEL_PATH,
# #         "test_samples": dice_test.shape[0],
# #         "Dice_Coefficient": {
# #             "RV": {"mean": float(df['dice_rv'].mean()), "std": float(df['dice_rv'].std())},
# #             "Myocardium": {"mean": float(df['dice_myo'].mean()), "std": float(df['dice_myo'].std())},
# #             "LV": {"mean": float(df['dice_lv'].mean()), "std": float(df['dice_lv'].std())},
# #             "Background": {"mean": float(df['dice_bg'].mean()), "std": float(df['dice_bg'].std())},
# #             "Mean_Foreground": {"mean": float(df['dice_mean'].mean()), "std": float(df['dice_mean'].std())}
# #         },
# #         "Hausdorff_Distance_95": {
# #             "RV": {"mean": float(df['hd95_rv'].mean()), "std": float(df['hd95_rv'].std()), "median": float(df['hd95_rv'].median())},
# #             "Myocardium": {"mean": float(df['hd95_myo'].mean()), "std": float(df['hd95_myo'].std()), "median": float(df['hd95_myo'].median())},
# #             "LV": {"mean": float(df['hd95_lv'].mean()), "std": float(df['hd95_lv'].std()), "median": float(df['hd95_lv'].median())},
# #             "Mean_Foreground": {"mean": float(df['hd95_mean'].mean()), "std": float(df['hd95_mean'].std())}
# #         }
# #     }

# #     json_path = os.path.join(RESULTS_DIR, "test_metrics_summary.json")
# #     with open(json_path, 'w') as f:
# #         json.dump(summary, f, indent=4)

# #     print("\n✓ 评估完成！详细结果保存至 CSV/JSON，并生成可视化图")
# #     return df, summary

# # # -------------------------------
# # # 运行评估
# # # -------------------------------
# # if __name__ == "__main__":
# #     df_results, summary = evaluate_model()

# import os
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import gc

# import segmentation_models as sm
# from segmentation_models.losses import dice_loss
# from segmentation_models.metrics import iou_score

# from keras.layers import Input, Conv2D
# from keras.models import Model
# from keras.optimizers import Adam
# from scipy.spatial.distance import directed_hausdorff  # 用于计算 Hausdorff 距离

# # ---------------------------
# # GPU 内存设置
# # ---------------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# path = ""

# # ---------------------------
# # 工具函数：指标计算
# # ---------------------------

# def calculate_dice(y_true, y_pred):
#     """计算单个样本的 Dice 系数 (不含背景层)"""
#     # y_true, y_pred 形状: (H, W)
#     dice_scores = []
#     for i in range(1, 4):  # 类别 1, 2, 3
#         gt = (y_true == i).astype(np.uint8)
#         pred = (y_pred == i).astype(np.uint8)
#         intersection = np.sum(gt * pred)
#         sum_val = np.sum(gt) + np.sum(pred)
#         if sum_val == 0:
#             dice_scores.append(1.0)
#         else:
#             dice_scores.append((2. * intersection) / sum_val)
#     return dice_scores

# def calculate_hausdorff(y_true, y_pred):
#     """计算单个样本的 Hausdorff 距离 (不含背景层)"""
#     hd_scores = []
#     for i in range(1, 4):  # 类别 1, 2, 3
#         gt_points = np.argwhere(y_true == i)
#         pred_points = np.argwhere(y_pred == i)
        
#         if len(gt_points) == 0 and len(pred_points) == 0:
#             hd_scores.append(0.0)
#         elif len(gt_points) == 0 or len(pred_points) == 0:
#             hd_scores.append(100.0)  # 缺失类别的惩罚值值
#         else:
#             # HD 是双向有向距离的最大值
#             d1 = directed_hausdorff(gt_points, pred_points)[0]
#             d2 = directed_hausdorff(pred_points, gt_points)[0]
#             hd_scores.append(max(d1, d2))
#     return hd_scores

# # ---------------------------
# # 工具函数：分批预测
# # ---------------------------
# def batch_predict(model, x_data, batch_size=4):
#     predictions = []
#     n_samples = x_data.shape[0]
#     for i in range(0, n_samples, batch_size):
#         end_idx = min(i + batch_size, n_samples)
#         batch_pred = model.predict(x_data[i:end_idx], verbose=0)
#         predictions.append(batch_pred)
#         tf.keras.backend.clear_session()
#         gc.collect()
#     return np.concatenate(predictions, axis=0)

# # ---------------------------
# # 加载与预处理数据
# # ---------------------------
# print("Loading data...")
# x_train = np.load(path + "x_2d_train.npy")
# y_train = np.load(path + "y_2d_train.npy")
# x_val = np.load(path + "x_2d_val.npy")
# y_val = np.load(path + "y_2d_val.npy")
# x_test = np.load(path + "x_2d_test.npy")
# y_test = np.load(path + "y_2d_test.npy")

# def normalize(x):
#     for i in range(x.shape[0]):
#         max_val = np.max(x[i, :, :, 0])
#         if max_val > 0:
#             x[i, :, :, 0] /= max_val
#     return x

# x_train, x_val, x_test = normalize(x_train), normalize(x_val), normalize(x_test)

# def create_onehot(y_data):
#     dim1, dim2, dim3, _ = y_data.shape
#     y_new = np.zeros((dim1, dim2, dim3, 4), dtype=np.float32)
#     for i in range(dim1):
#         y_new[i, :, :, 0] = (y_data[i, :, :, 0] == 1)
#         y_new[i, :, :, 1] = (y_data[i, :, :, 0] == 2)
#         y_new[i, :, :, 2] = (y_data[i, :, :, 0] == 3)
#         y_new[i, :, :, 3] = (y_data[i, :, :, 0] == 0)
#     return y_new

# y_train_oh = create_onehot(y_train)
# y_val_oh = create_onehot(y_val)
# y_test_oh = create_onehot(y_test)

# # ---------------------------
# # 构建与加载模型
# # ---------------------------
# print("Building model...")
# base_model = sm.Unet(backbone_name="vgg16", input_shape=(256, 256, 3), classes=4, activation="softmax", encoder_weights="imagenet")
# inp = Input(shape=(None, None, x_train.shape[-1]))
# l1 = Conv2D(3, (1, 1))(inp)
# out = base_model(l1)
# modelUnet = Model(inp, out, name=base_model.name)

# # 这里的 sm.metrics.FScore(beta=1) 本质上就是 Dice 系数
# metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
# modelUnet.compile(Adam(0.0001), sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(), metrics)

# print("Loading weights...")
# modelUnet.load_weights("risultati100epochs/modelUnet_50epochs.keras")

# # ---------------------------
# # 预测与评估
# # ---------------------------
# print("Evaluating on Test Set...")
# # 预测测试集前 50 个样本（HD计算较慢，可根据需要调整数量）
# num_eval = min(len(x_test), 50) 
# y_pred_test = batch_predict(modelUnet, x_test[:num_eval], batch_size=2)

# def back_to_1_channel_mask(img, alpha=0.5):
#     yy = np.zeros((img.shape[0], 256, 256))
#     yy += 1.0 * (img[:, :, :, 0] >= alpha)
#     yy += 2.0 * (img[:, :, :, 1] >= alpha)
#     yy += 3.0 * (img[:, :, :, 2] >= alpha)
#     return yy

# # 计算 Dice 和 Hausdorff
# all_dice = []
# all_hd = []

# for i in range(num_eval):
#     # 将 one-hot 转回单通道类别索引 (0, 1, 2, 3)
#     true_mask = back_to_1_channel_mask(y_test_oh[i:i+1], 0.5)[0]
#     pred_mask = back_to_1_channel_mask(y_pred_test[i:i+1], 0.5)[0]
    
#     all_dice.append(calculate_dice(true_mask, pred_mask))
#     all_hd.append(calculate_hausdorff(true_mask, pred_mask))

# avg_dice = np.mean(all_dice, axis=0)
# avg_hd = np.mean(all_hd, axis=0)

# print("\n" + "="*40)
# print("TEST SET METRICS (Excluding Background):")
# for i in range(3):
#     print(f"Class {i+1} -> Dice: {avg_dice[i]:.4f}, Hausdorff Distance: {avg_hd[i]:.4f}")
# print("="*40 + "\n")

# # ---------------------------
# # 生成对比图 (保持原逻辑并保存)
# # ---------------------------
# os.makedirs("plot", exist_ok=True)
# print("Generating plots...")
# for i in range(min(num_eval, 5)):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     true_mask = back_to_1_channel_mask(y_test_oh[i:i+1], 0.5)[0]
#     pred_mask = back_to_1_channel_mask(y_pred_test[i:i+1], 0.5)[0]
    
#     axes[0].imshow(x_test[i, :, :, 0], cmap='gray')
#     axes[0].set_title(f"Test Image {i}")
    
#     axes[1].imshow(x_test[i, :, :, 0], cmap='gray')
#     axes[1].imshow(true_mask, cmap='jet', alpha=0.5)
#     axes[1].set_title("True Mask")
    
#     axes[2].imshow(x_test[i, :, :, 0], cmap='gray')
#     axes[2].imshow(pred_mask, cmap='jet', alpha=0.5)
#     axes[2].set_title(f"Pred (Dice: {np.mean(all_dice[i]):.2f})")
    
#     for ax in axes: ax.axis('off')
#     plt.savefig(f"plot/eval_{i}.png")
#     plt.close()

# print("Process completed.")

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

