import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from loguru import logger
# -------------------------------
# 工具函数
# -------------------------------

def normalize_volume(img_data):
    """3D 层面进行 Z-Score 归一化"""
    p99 = np.percentile(img_data, 99)
    p1 = np.percentile(img_data, 1)
    img_data = np.clip(img_data, p1, p99)
    
    mean = np.mean(img_data)
    std = np.std(img_data)
    return (img_data - mean) / (std + 1e-8)

def center_padding_or_crop(img, target_size=(256, 256)):
    """中心填充或中心裁剪到 256x256"""
    if len(img.shape) == 3:
        h, w, d = img.shape
    else:
        # 处理可能的维度异常
        return img

    new_image = np.zeros((target_size[0], target_size[1], d))
    
    # 计算目标位置（填充时）
    start_h = max(0, (target_size[0] - h) // 2)
    start_w = max(0, (target_size[1] - w) // 2)
    
    # 计算源位置（裁剪时）
    src_start_h = max(0, (h - target_size[0]) // 2)
    src_start_w = max(0, (w - target_size[1]) // 2)
    
    # 实际拷贝尺寸
    copy_h = min(h, target_size[0])
    copy_w = min(w, target_size[1])
    
    new_image[start_h:start_h+copy_h, start_w:start_w+copy_w, :] = \
        img[src_start_h:src_start_h+copy_h, src_start_w:src_start_w+copy_w, :]
    
    return new_image

def process_patient_list(p_list, data_path):
    """处理病人列表，返回 2D 切片数据"""
    x_list, y_list = [], []
    for pid in p_list:
        p_path = os.path.join(data_path, pid)
        # 查找该病人目录下所有的 .nii.gz 文件
        files = [f for f in os.listdir(p_path) if f.endswith('.nii.gz')]
        
        # 过滤出标签文件 (含有 _gt)
        gt_files = [f for f in files if '_gt' in f]
        
        for gt_f in gt_files:
            img_f = gt_f.replace('_gt', '')
            img_full_path = os.path.join(p_path, img_f)
            gt_full_path = os.path.join(p_path, gt_f)
            
            if not os.path.exists(img_full_path):
                continue
                
            # 加载 nibabel 对象
            img_nii = nib.load(img_full_path)
            gt_nii = nib.load(gt_full_path)
            
            img_data = img_nii.get_fdata()
            gt_data = gt_nii.get_fdata()
            
            # 1. 强度归一化 (仅针对原图)
            img_data = normalize_volume(img_data)
            
            # 2. 空间对齐 (中心填充/裁剪)
            img_data = center_padding_or_crop(img_data)
            gt_data = center_padding_or_crop(gt_data)
            
            # 3. 展开为 2D 切片
            for slice_idx in range(img_data.shape[2]):
                x_list.append(img_data[:, :, slice_idx])
                y_list.append(gt_data[:, :, slice_idx])
                
    # 转换为 numpy 数组并增加通道维度 (N, H, W, 1)
    if len(x_list) == 0:
        return np.array([]), np.array([])
    return np.expand_dims(np.array(x_list), -1), np.expand_dims(np.array(y_list), -1)

# -------------------------------
# 主函数
# -------------------------------

def load_data(train_data_folder, save_data_folder):
    """
    加载、预处理并保存医学影像数据
    
    参数:
        train_data_folder: 训练数据文件夹路径，应包含 patient* 子文件夹
        save_data_folder: 保存处理后 .npy 文件的文件夹路径
    """
    # 创建保存文件夹（如果不存在）
    os.makedirs(save_data_folder, exist_ok=True)
    
    # -------------------------------
    # 数据加载与划分
    # -------------------------------
    
    data_path = train_data_folder
    # 获取所有 patient 文件夹名
    patient_ids = sorted([d for d in os.listdir(data_path) if d.startswith('patient')])
    
    # 严格按病人划分数据集，防止数据泄露 (Data Leakage)
    train_ids, test_val_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)
    
    # -------------------------------
    # 执行与保存
    # -------------------------------
    
    logger.info(f"Total patients: {len(patient_ids)}")
    logger.info("Processing training data...")
    x_train, y_train = process_patient_list(train_ids, data_path)
    
    logger.info("Processing validation data...")
    x_val, y_val = process_patient_list(val_ids, data_path)
    
    logger.info("Processing test data...")
    x_test, y_test = process_patient_list(test_ids, data_path)
    
    # 保存
    np.save(os.path.join(save_data_folder, "x_2d_train.npy"), x_train.astype(np.float32))
    np.save(os.path.join(save_data_folder, "y_2d_train.npy"), y_train.astype(np.float32))
    np.save(os.path.join(save_data_folder, "x_2d_val.npy"), x_val.astype(np.float32))
    np.save(os.path.join(save_data_folder, "y_2d_val.npy"), y_val.astype(np.float32))
    np.save(os.path.join(save_data_folder, "x_2d_test.npy"), x_test.astype(np.float32))
    np.save(os.path.join(save_data_folder, "y_2d_test.npy"), y_test.astype(np.float32))
    
    logger.info("-" * 30)
    logger.info(f"Train slices: {x_train.shape[0]}")
    logger.info(f"Val slices:   {x_val.shape[0]}")
    logger.info(f"Test slices:  {x_test.shape[0]}")
    logger.info(f"All .npy files saved to: {save_data_folder}")

FILE_NAMES = [
    "x_2d_train.npy", "y_2d_train.npy",
    "x_2d_val.npy", "y_2d_val.npy",
    "x_2d_test.npy", "y_2d_test.npy"
]

def get_npy_file_paths(save_data_folder):
    """
    返回保存的六个文件的完整路径列表
    """
    return [os.path.join(save_data_folder, f) for f in FILE_NAMES]

def check_npy_files_exist(save_data_folder):
    """
    检查 save_folder 下是否存在这六个文件
    返回：布尔值（全部存在为 True，否则为 False）和 缺失文件列表
    """
    missing_files = []
    for f in FILE_NAMES:
        path = os.path.join(save_data_folder, f)
        if not os.path.exists(path):
            missing_files.append(f)
    
    is_complete = len(missing_files) == 0
    return is_complete, missing_files