# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# evaluate_model(weights_path, result_dir, ...)
# Memory-efficient evaluation wrapper that computes per-class Dice and HD95 (LV/RV/MYO assumed labels 1/2/3).
# Saves CSVs and PNGs into result_dir and returns summary DataFrames.
# """
# import os
# import gc
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # optional tqdm/ scipy
# try:
#     from tqdm import tqdm
# except Exception:
#     tqdm = lambda x, **kw: x

# try:
#     import scipy.ndimage as ndi
#     from scipy.ndimage import binary_erosion
#     SCIPY_AVAILABLE = True
# except Exception:
#     SCIPY_AVAILABLE = False
#     try:
#         from scipy.spatial.distance import directed_hausdorff
#     except Exception:
#         directed_hausdorff = None

# # reduce TF logs
# os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# # --- Helper functions ---
# def _set_gpu_memory_growth():
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print("GPU memory growth not set:", e)

# def safe_normalize_batch(x_batch):
#     x_batch = x_batch.astype(np.float32)
#     maxvals = x_batch[..., 0].max(axis=(1,2), keepdims=True)  # (B,1,1)
#     maxvals_b = maxvals.copy()
#     maxvals_b[maxvals_b == 0] = 1.0
#     x_batch[..., 0] = x_batch[..., 0] / maxvals_b.squeeze(axis=(1,2))
#     return x_batch

# def to_one_hot_single(y_batch, num_classes):
#     y = y_batch
#     if y.ndim == 4 and y.shape[-1] == 1:
#         y = y[..., 0]
#     return tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)

# def dice_binary(pred, gt, eps=1e-6):
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     inter = np.logical_and(pred, gt).sum()
#     denom = pred.sum() + gt.sum()
#     if denom == 0:
#         return 1.0
#     return 2.0 * inter / (denom + eps)

# def hd95_binary(pred, gt, voxel_spacing=1.0):
#     pred = pred.astype(bool)
#     gt = gt.astype(bool)
#     # both empty
#     if not pred.any() and not gt.any():
#         return 0.0
#     if not pred.any() or not gt.any():
#         return np.inf
#     if SCIPY_AVAILABLE:
#         dt_pred = ndi.distance_transform_edt(~pred) * voxel_spacing
#         dt_gt = ndi.distance_transform_edt(~gt) * voxel_spacing
#         pred_eroded = binary_erosion(pred)
#         gt_eroded = binary_erosion(gt)
#         surf_pred = np.logical_xor(pred, pred_eroded)
#         surf_gt = np.logical_xor(gt, gt_eroded)
#         d_pred_to_gt = dt_gt[surf_pred].ravel()
#         d_gt_to_pred = dt_pred[surf_gt].ravel()
#         a = d_pred_to_gt if d_pred_to_gt.size>0 else np.array([0.0])
#         b = d_gt_to_pred if d_gt_to_pred.size>0 else np.array([0.0])
#         all_d = np.concatenate([a, b])
#         return float(np.percentile(all_d, 95))
#     else:
#         # fallback approximate (may be slower / less precise)
#         if directed_hausdorff is None:
#             return np.inf
#         coords_pred = np.column_stack(np.where(pred))
#         coords_gt = np.column_stack(np.where(gt))
#         if coords_pred.size == 0 or coords_gt.size == 0:
#             return np.inf
#         d1 = directed_hausdorff(coords_pred, coords_gt)[0]
#         d2 = directed_hausdorff(coords_gt, coords_pred)[0]
#         return float(max(d1, d2) * voxel_spacing)

# def data_generator(path_prefix, prefix, batch_size=1, shuffle=False, yield_labels=True):
#     x_path = os.path.join(path_prefix, f"x_2d_{prefix}.npy")
#     y_path = os.path.join(path_prefix, f"y_2d_{prefix}.npy")
#     x_mm = np.load(x_path, mmap_mode='r')
#     y_mm = np.load(y_path, mmap_mode='r')
#     n = x_mm.shape[0]
#     idx = np.arange(n)
#     if shuffle:
#         np.random.shuffle(idx)
#     for start in range(0, n, batch_size):
#         batch_idx = idx[start:start+batch_size]
#         x_batch = np.array([x_mm[i] for i in batch_idx], dtype=np.float32)
#         x_batch = safe_normalize_batch(x_batch)
#         if yield_labels:
#             y_batch = np.array([y_mm[i] for i in batch_idx], dtype=np.int32)
#             y_onehot = to_one_hot_single(y_batch, num_classes=4)
#             yield x_batch, y_onehot
#         else:
#             yield x_batch

# def dataset_for_tf(path_prefix, prefix, batch_size=1, yield_labels=True):
#     gen = lambda: data_generator(path_prefix, prefix, batch_size=batch_size, shuffle=False, yield_labels=yield_labels)
#     if yield_labels:
#         out_sig = (
#             tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
#             tf.TensorSpec(shape=(None, None, None, 4), dtype=tf.float32),
#         )
#     else:
#         out_sig = tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
#     ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
#     return ds.prefetch(tf.data.AUTOTUNE)

# def get_data_shape(path_prefix, prefix):
#     x_path = os.path.join(path_prefix, f"x_2d_{prefix}.npy")
#     x_mm = np.load(x_path, mmap_mode='r')
#     return x_mm.shape

# # --- Main evaluate_model function ---
# def evaluate_model(WEIGHTS_PATH,
#                    data_path,
#                    RESULT_DIR="result/evaluation/evaluation_results",
#                    batch_size=1,
#                    eval_only_test=True,
#                    voxel_spacing=1.0,
#                    class_labels=None,
#                    backbone="vgg16"):
#     """
#     Run memory-efficient evaluation and save results to RESULT_DIR.
#     Returns dict with DataFrames: summary_df, agg_df, per_sample_df and result_dir path.
#     """
#     os.makedirs(RESULT_DIR, exist_ok=True)
#     if class_labels is None:
#         class_labels = {1: "LV", 2: "RV", 3: "MYO", 0: "Background"}

#     _set_gpu_memory_growth()
#     tf.keras.backend.clear_session()

#     # import segmentation_models inside function (so errors are clearer)
#     import segmentation_models as sm
#     from keras.layers import Input, Conv2D
#     from keras.models import Model
#     from keras.optimizers import Adam

#     print("Reading test shape (memmap)...")
#     test_shape = get_data_shape(data_path, "test")
#     n_channels = test_shape[-1]

#     print("Building model...")
#     base_model = sm.Unet(backbone_name=backbone, input_shape=(256,256,3), classes=4, activation='softmax', encoder_weights='imagenet')
#     inp = Input(shape=(None, None, n_channels))
#     l1 = Conv2D(3, (1,1))(inp)
#     out = base_model(l1)
#     modelUnet = Model(inp, out, name=base_model.name)

#     optim = Adam(learning_rate=1e-4)
#     dice_loss = sm.losses.DiceLoss(class_weights=np.array([1,1,1,0.5], dtype=np.float32))
#     focal_loss = sm.losses.CategoricalFocalLoss()
#     total_loss = dice_loss + focal_loss
#     metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
#     modelUnet.compile(optimizer=optim, loss=total_loss, metrics=metrics)

#     print("Loading weights:", WEIGHTS_PATH)
#     modelUnet.load_weights(WEIGHTS_PATH)
#     print("Weights loaded.")

#     # Evaluate via tf.data dataset (memory efficient)
#     if not eval_only_test:
#         ds_train = dataset_for_tf(data_path, "train", batch_size=batch_size, yield_labels=True)
#         train_metrics = modelUnet.evaluate(ds_train, verbose=1)
#         ds_val = dataset_for_tf(data_path, "val", batch_size=batch_size, yield_labels=True)
#         val_metrics = modelUnet.evaluate(ds_val, verbose=1)
#     else:
#         train_metrics = val_metrics = None

#     ds_test = dataset_for_tf(data_path, "test", batch_size=batch_size, yield_labels=True)
#     test_metrics = modelUnet.evaluate(ds_test, verbose=1)

#     # Predict batch-by-batch and compute per-sample metrics
#     pred_gen = data_generator(data_path, "test", batch_size=batch_size, shuffle=False, yield_labels=True)
#     n_samples = test_shape[0]
#     per_sample_records = []
#     dice_per_class_list = {cls: [] for cls in range(4)}
#     hd95_per_class_list = {cls: [] for cls in range(4)}

#     sample_idx = 0
#     it = tqdm(pred_gen, total=(n_samples // max(1, batch_size)))
#     for x_batch, y_batch_onehot in it:
#         y_pred_batch = modelUnet.predict(x_batch, verbose=0)
#         y_pred_labels = np.argmax(y_pred_batch, axis=-1)
#         y_true_labels = np.argmax(y_batch_onehot, axis=-1)
#         B = y_pred_labels.shape[0]
#         for b in range(B):
#             pred = y_pred_labels[b]
#             true = y_true_labels[b]
#             rec = {'sample': sample_idx}
#             for cls in range(4):
#                 pred_bin = (pred == cls)
#                 true_bin = (true == cls)
#                 dval = dice_binary(pred_bin, true_bin)
#                 if cls == 0:
#                     hd = np.nan
#                 else:
#                     hd = hd95_binary(pred_bin, true_bin, voxel_spacing=voxel_spacing)
#                     if np.isinf(hd):
#                         hd = np.nan
#                 rec[f"class_{cls}_dice"] = float(dval)
#                 rec[f"class_{cls}_hd95"] = float(hd) if not np.isnan(hd) else np.nan
#                 dice_per_class_list[cls].append(dval)
#                 hd95_per_class_list[cls].append(hd)
#             rec['mean_dice_lv_rv_myo'] = float(np.nanmean([rec['class_1_dice'], rec['class_2_dice'], rec['class_3_dice']]))
#             per_sample_records.append(rec)
#             sample_idx += 1
#         del x_batch, y_batch_onehot, y_pred_batch, y_pred_labels, y_true_labels
#         gc.collect()

#     per_sample_df = pd.DataFrame(per_sample_records)
#     per_sample_csv = os.path.join(RESULT_DIR, "per_sample_metrics.csv")
#     per_sample_df.to_csv(per_sample_csv, index=False)

#     agg_rows = []
#     for cls in range(4):
#         dice_arr = np.array(dice_per_class_list[cls], dtype=np.float32)
#         hd_arr = np.array(hd95_per_class_list[cls], dtype=np.float32)
#         dice_mean = float(np.nanmean(dice_arr))
#         dice_std = float(np.nanstd(dice_arr))
#         hd_mean = float(np.nanmean(hd_arr[np.isfinite(hd_arr)])) if np.any(np.isfinite(hd_arr)) else np.nan
#         hd_median = float(np.nanmedian(hd_arr[np.isfinite(hd_arr)])) if np.any(np.isfinite(hd_arr)) else np.nan
#         agg_rows.append({
#             "class": class_labels.get(cls, f"Class_{cls}"),
#             "label": int(cls),
#             "dice_mean": dice_mean,
#             "dice_std": dice_std,
#             "hd95_mean": hd_mean,
#             "hd95_median": hd_median
#         })
#     agg_df = pd.DataFrame(agg_rows)
#     agg_csv = os.path.join(RESULT_DIR, "per_class_summary.csv")
#     agg_df.to_csv(agg_csv, index=False)

#     # Combined summary
#     summary = {
#         "Dataset": ["Test"],
#         "Loss": [float(test_metrics[0])],
#         "IoU_Score": [float(test_metrics[1])],
#         "F_Score": [float(test_metrics[2])],
#         "LV_Dice": [agg_df.loc[agg_df['label']==1,'dice_mean'].values[0]],
#         "RV_Dice": [agg_df.loc[agg_df['label']==2,'dice_mean'].values[0]],
#         "MYO_Dice":[agg_df.loc[agg_df['label']==3,'dice_mean'].values[0]],
#         "LV_HD95": [agg_df.loc[agg_df['label']==1,'hd95_mean'].values[0]],
#         "RV_HD95": [agg_df.loc[agg_df['label']==2,'hd95_mean'].values[0]],
#         "MYO_HD95":[agg_df.loc[agg_df['label']==3,'hd95_mean'].values[0]],
#     }
#     summary_df = pd.DataFrame(summary)
#     summary_csv = os.path.join(RESULT_DIR, "evaluation_summary.csv")
#     summary_df.to_csv(summary_csv, index=False)

#     # Visuals
#     labels = [class_labels[1], class_labels[2], class_labels[3]]
#     dice_means = [agg_df.loc[agg_df['label']==i,'dice_mean'].values[0] for i in [1,2,3]]
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.bar(np.arange(len(labels)), dice_means)
#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)
#     ax.set_ylim(0,1)
#     ax.set_ylabel("Dice")
#     ax.set_title("Mean Dice by Class (Test)")
#     plt.tight_layout()
#     dice_png = os.path.join(RESULT_DIR, "dice_by_class.png")
#     plt.savefig(dice_png, dpi=150); plt.close()

#     hd_means = [agg_df.loc[agg_df['label']==i,'hd95_mean'].values[0] for i in [1,2,3]]
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.bar(np.arange(len(labels)), hd_means)
#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)
#     ax.set_ylabel("HD95 (pixels)")
#     ax.set_title("Mean HD95 by Class (Test)")
#     plt.tight_layout()
#     hd_png = os.path.join(RESULT_DIR, "hd95_by_class.png")
#     plt.savefig(hd_png, dpi=150); plt.close()

#     print("Saved:", per_sample_csv, agg_csv, summary_csv, dice_png, hd_png)

#     # Return dataframes and paths for further programmatic use
#     return {
#         "summary_df": summary_df,
#         "per_class_df": agg_df,
#         "per_sample_df": per_sample_df,
#         "result_dir": RESULT_DIR
#     }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_model(weights_path, result_dir, ...)
Memory-efficient evaluation wrapper that computes per-class Dice and HD95 (LV/RV/MYO assumed labels 1/2/3).
Saves CSVs and PNGs into result_dir and returns summary DataFrames.
"""
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger

# optional tqdm/ scipy
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x

try:
    import scipy.ndimage as ndi
    from scipy.ndimage import binary_erosion
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    try:
        from scipy.spatial.distance import directed_hausdorff
    except Exception:
        directed_hausdorff = None

# reduce TF logs
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# --- Helper functions ---
def _set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU memory growth not set: {e}")

def safe_normalize_batch(x_batch):
    x_batch = x_batch.astype(np.float32)
    maxvals = x_batch[..., 0].max(axis=(1,2), keepdims=True)  # (B,1,1)
    maxvals_b = maxvals.copy()
    maxvals_b[maxvals_b == 0] = 1.0
    x_batch[..., 0] = x_batch[..., 0] / maxvals_b.squeeze(axis=(1,2))
    return x_batch

def to_one_hot_single(y_batch, num_classes):
    y = y_batch
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y[..., 0]
    return tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)

def dice_binary(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / (denom + eps)

def hd95_binary(pred, gt, voxel_spacing=1.0):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    # both empty
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.inf
    if SCIPY_AVAILABLE:
        dt_pred = ndi.distance_transform_edt(~pred) * voxel_spacing
        dt_gt = ndi.distance_transform_edt(~gt) * voxel_spacing
        pred_eroded = binary_erosion(pred)
        gt_eroded = binary_erosion(gt)
        surf_pred = np.logical_xor(pred, pred_eroded)
        surf_gt = np.logical_xor(gt, gt_eroded)
        d_pred_to_gt = dt_gt[surf_pred].ravel()
        d_gt_to_pred = dt_pred[surf_gt].ravel()
        a = d_pred_to_gt if d_pred_to_gt.size>0 else np.array([0.0])
        b = d_gt_to_pred if d_gt_to_pred.size>0 else np.array([0.0])
        all_d = np.concatenate([a, b])
        return float(np.percentile(all_d, 95))
    else:
        # fallback approximate (may be slower / less precise)
        if directed_hausdorff is None:
            return np.inf
        coords_pred = np.column_stack(np.where(pred))
        coords_gt = np.column_stack(np.where(gt))
        if coords_pred.size == 0 or coords_gt.size == 0:
            return np.inf
        d1 = directed_hausdorff(coords_pred, coords_gt)[0]
        d2 = directed_hausdorff(coords_gt, coords_pred)[0]
        return float(max(d1, d2) * voxel_spacing)

def data_generator(path_prefix, prefix, batch_size=1, shuffle=False, yield_labels=True):
    x_path = os.path.join(path_prefix, f"x_2d_{prefix}.npy")
    y_path = os.path.join(path_prefix, f"y_2d_{prefix}.npy")
    x_mm = np.load(x_path, mmap_mode='r')
    y_mm = np.load(y_path, mmap_mode='r')
    n = x_mm.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start+batch_size]
        x_batch = np.array([x_mm[i] for i in batch_idx], dtype=np.float32)
        x_batch = safe_normalize_batch(x_batch)
        if yield_labels:
            y_batch = np.array([y_mm[i] for i in batch_idx], dtype=np.int32)
            y_onehot = to_one_hot_single(y_batch, num_classes=4)
            yield x_batch, y_onehot
        else:
            yield x_batch

def dataset_for_tf(path_prefix, prefix, batch_size=1, yield_labels=True):
    gen = lambda: data_generator(path_prefix, prefix, batch_size=batch_size, shuffle=False, yield_labels=yield_labels)
    if yield_labels:
        out_sig = (
            tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, 4), dtype=tf.float32),
        )
    else:
        out_sig = tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    return ds.prefetch(tf.data.AUTOTUNE)

def get_data_shape(path_prefix, prefix):
    x_path = os.path.join(path_prefix, f"x_2d_{prefix}.npy")
    x_mm = np.load(x_path, mmap_mode='r')
    return x_mm.shape

# --- Main evaluate_model function ---
def evaluate_model(WEIGHTS_PATH,
                   data_path,
                   RESULT_DIR="result/evaluation/evaluation_results",
                   batch_size=1,
                   eval_only_test=True,
                   voxel_spacing=1.0,
                   class_labels=None,
                   backbone="vgg16"):
    os.makedirs(RESULT_DIR, exist_ok=True)
    if class_labels is None:
        class_labels = {1: "LV", 2: "RV", 3: "MYO", 0: "Background"}

    _set_gpu_memory_growth()
    tf.keras.backend.clear_session()

    import segmentation_models as sm
    from keras.layers import Input, Conv2D
    from keras.models import Model
    from keras.optimizers import Adam

    logger.info("Reading test shape (memmap)...")
    test_shape = get_data_shape(data_path, "test")
    n_channels = test_shape[-1]
    logger.info(f"Test data shape: {test_shape}, number of channels: {n_channels}")

    logger.info("Building model...")
    base_model = sm.Unet(backbone_name=backbone, input_shape=(256,256,3), classes=4, activation='softmax', encoder_weights='imagenet')
    inp = Input(shape=(None, None, n_channels))
    l1 = Conv2D(3, (1,1))(inp)
    out = base_model(l1)
    modelUnet = Model(inp, out, name=base_model.name)

    optim = Adam(learning_rate=1e-4)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1,1,1,0.5], dtype=np.float32))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + focal_loss
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    modelUnet.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    logger.info(f"Loading weights from: {WEIGHTS_PATH}")
    modelUnet.load_weights(WEIGHTS_PATH)
    logger.info("Weights loaded successfully.")

    if not eval_only_test:
        ds_train = dataset_for_tf(data_path, "train", batch_size=batch_size, yield_labels=True)
        train_metrics = modelUnet.evaluate(ds_train, verbose=1)
        ds_val = dataset_for_tf(data_path, "val", batch_size=batch_size, yield_labels=True)
        val_metrics = modelUnet.evaluate(ds_val, verbose=1)
    else:
        train_metrics = val_metrics = None

    ds_test = dataset_for_tf(data_path, "test", batch_size=batch_size, yield_labels=True)
    test_metrics = modelUnet.evaluate(ds_test, verbose=1)
    logger.info(f"Test evaluation metrics: Loss={test_metrics[0]}, IoU={test_metrics[1]}, FScore={test_metrics[2]}")

    logger.info("Starting batch prediction and per-sample metric computation...")
    pred_gen = data_generator(data_path, "test", batch_size=batch_size, shuffle=False, yield_labels=True)
    n_samples = test_shape[0]
    per_sample_records = []
    dice_per_class_list = {cls: [] for cls in range(4)}
    hd95_per_class_list = {cls: [] for cls in range(4)}

    sample_idx = 0
    it = tqdm(pred_gen, total=(n_samples // max(1, batch_size)))
    for x_batch, y_batch_onehot in it:
        y_pred_batch = modelUnet.predict(x_batch, verbose=0)
        y_pred_labels = np.argmax(y_pred_batch, axis=-1)
        y_true_labels = np.argmax(y_batch_onehot, axis=-1)
        B = y_pred_labels.shape[0]
        for b in range(B):
            pred = y_pred_labels[b]
            true = y_true_labels[b]
            rec = {'sample': sample_idx}
            for cls in range(4):
                pred_bin = (pred == cls)
                true_bin = (true == cls)
                dval = dice_binary(pred_bin, true_bin)
                if cls == 0:
                    hd = np.nan
                else:
                    hd = hd95_binary(pred_bin, true_bin, voxel_spacing=voxel_spacing)
                    if np.isinf(hd):
                        hd = np.nan
                rec[f"class_{cls}_dice"] = float(dval)
                rec[f"class_{cls}_hd95"] = float(hd) if not np.isnan(hd) else np.nan
                dice_per_class_list[cls].append(dval)
                hd95_per_class_list[cls].append(hd)
            rec['mean_dice_lv_rv_myo'] = float(np.nanmean([rec['class_1_dice'], rec['class_2_dice'], rec['class_3_dice']]))
            per_sample_records.append(rec)
            sample_idx += 1
        del x_batch, y_batch_onehot, y_pred_batch, y_pred_labels, y_true_labels
        gc.collect()

    per_sample_df = pd.DataFrame(per_sample_records)
    per_sample_csv = os.path.join(RESULT_DIR, "per_sample_metrics.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)
    logger.info(f"Saved per-sample metrics to: {per_sample_csv}")

    agg_rows = []
    for cls in range(4):
        dice_arr = np.array(dice_per_class_list[cls], dtype=np.float32)
        hd_arr = np.array(hd95_per_class_list[cls], dtype=np.float32)
        dice_mean = float(np.nanmean(dice_arr))
        dice_std = float(np.nanstd(dice_arr))
        hd_mean = float(np.nanmean(hd_arr[np.isfinite(hd_arr)])) if np.any(np.isfinite(hd_arr)) else np.nan
        hd_median = float(np.nanmedian(hd_arr[np.isfinite(hd_arr)])) if np.any(np.isfinite(hd_arr)) else np.nan
        agg_rows.append({
            "class": class_labels.get(cls, f"Class_{cls}"),
            "label": int(cls),
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "hd95_mean": hd_mean,
            "hd95_median": hd_median
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(RESULT_DIR, "per_class_summary.csv")
    agg_df.to_csv(agg_csv, index=False)
    logger.info(f"Saved per-class summary to: {agg_csv}")

    summary = {
        "Dataset": ["Test"],
        "Loss": [float(test_metrics[0])],
        "IoU_Score": [float(test_metrics[1])],
        "F_Score": [float(test_metrics[2])],
        "LV_Dice": [agg_df.loc[agg_df['label']==1,'dice_mean'].values[0]],
        "RV_Dice": [agg_df.loc[agg_df['label']==2,'dice_mean'].values[0]],
        "MYO_Dice":[agg_df.loc[agg_df['label']==3,'dice_mean'].values[0]],
        "LV_HD95": [agg_df.loc[agg_df['label']==1,'hd95_mean'].values[0]],
        "RV_HD95": [agg_df.loc[agg_df['label']==2,'hd95_mean'].values[0]],
        "MYO_HD95":[agg_df.loc[agg_df['label']==3,'hd95_mean'].values[0]],
    }
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(RESULT_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved overall summary to: {summary_csv}")

    labels = [class_labels[1], class_labels[2], class_labels[3]]
    dice_means = [agg_df.loc[agg_df['label']==i,'dice_mean'].values[0] for i in [1,2,3]]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(np.arange(len(labels)), dice_means)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_ylabel("Dice")
    ax.set_title("Mean Dice by Class (Test)")
    plt.tight_layout()
    dice_png = os.path.join(RESULT_DIR, "dice_by_class.png")
    plt.savefig(dice_png, dpi=150); plt.close()
    logger.info(f"Saved Dice plot to: {dice_png}")

    hd_means = [agg_df.loc[agg_df['label']==i,'hd95_mean'].values[0] for i in [1,2,3]]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(np.arange(len(labels)), hd_means)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("HD95 (pixels)")
    ax.set_title("Mean HD95 by Class (Test)")
    plt.tight_layout()
    hd_png = os.path.join(RESULT_DIR, "hd95_by_class.png")
    plt.savefig(hd_png, dpi=150); plt.close()
    logger.info(f"Saved HD95 plot to: {hd_png}")

    logger.info("Evaluation completed successfully.")

    return {
        "summary_df": summary_df,
        "per_class_df": agg_df,
        "per_sample_df": per_sample_df,
        "result_dir": RESULT_DIR
    }
