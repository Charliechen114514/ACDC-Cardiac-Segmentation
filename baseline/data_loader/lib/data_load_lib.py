import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import time
from loguru import logger

"""
Medical Image Data Loader Module
用于加载和处理医学图像训练数据的模块
"""

import os
import fnmatch
import numpy as np
import nibabel as nib


# ==================== 工具函数 ====================

def find(pattern, path):
    """
    在指定路径中查找匹配模式的文件
    
    Args:
        pattern: 文件名匹配模式
        path: 搜索路径
    
    Returns:
        匹配文件的完整路径列表
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def padding_image(img):
    """
    将图像填充到256x256大小
    
    Args:
        img: 输入图像数组
    
    Returns:
        填充后的图像数组
    """
    (dim1, dim2, dim3) = img.shape
    new_image = np.zeros(shape=(256, 256, dim3))
    new_image[0:dim1, 0:dim2, 0:dim3] = img
    return new_image


def str_fun(i):
    """
    将数字格式化为3位字符串（补零）
    
    Args:
        i: 输入数字
    
    Returns:
        格式化后的字符串（例如：1->"001", 10->"010", 100->"100"）
    """
    if i <= 9:
        return "00" + str(i)
    if i <= 99:
        return "0" + str(i)
    return str(i)


# ==================== 图像加载函数 ====================

def load_image_if_valid(file_path):
    """
    加载图像并检查尺寸是否有效（最大维度<257）
    
    Args:
        file_path: 图像文件路径
    
    Returns:
        填充后的图像数据，如果尺寸无效则返回None
    """
    img = nib.load(file_path)
    if np.max(img.shape) < 257:
        return padding_image(img.get_fdata())
    return None


def process_file_list(file_list, x_train, y_train):
    """
    处理文件列表，将图像数据添加到训练集
    
    Args:
        file_list: 待处理的文件路径列表
        x_train: 训练图像列表（会被修改）
        y_train: 标签图像列表（会被修改）
    """
    for name in file_list:
        if fnmatch.fnmatch(name, "*_gt.nii.gz"):
            # 这是ground truth标签文件
            padded_img = load_image_if_valid(name)
            if padded_img is not None:
                y_train.append(padded_img)
        else:
            # 这是训练图像文件
            padded_img = load_image_if_valid(name)
            if padded_img is not None:
                x_train.append(padded_img)


def process_patient_data(patient_num, base_path, x_train, y_train):
    """
    处理单个病人的所有图像数据
    
    Args:
        patient_num: 病人编号（1-100）
        base_path: 数据基础路径
        x_train: 训练图像列表（会被修改）
        y_train: 标签图像列表（会被修改）
    """
    # 构建病人数据路径
    patient_id = "patient" + str_fun(patient_num)
    patient_path = os.path.join(base_path, patient_id)

    if not os.path.exists(patient_path):
        raise FileNotFoundError(f"Patient path does not exist: {patient_path}")

    patient_pattern = patient_id + "_frame*"
    
    # 查找所有匹配的文件
    all_files = find(patient_pattern, patient_path)
    
    # 分离frame01和其他帧的文件
    frame01_files = []
    other_frame_files = []
    
    for name in all_files:
        if fnmatch.fnmatch(name, "*frame01"):
            frame01_files.append(name)
        else:
            other_frame_files.append(name)
    
    # 先处理frame01的文件
    process_file_list(frame01_files, x_train, y_train)
    
    # 再处理其他帧的文件
    process_file_list(other_frame_files, x_train, y_train)


# ==================== 主加载函数 ====================

def load_training_data(base_path, num_patients=100):
    """
    加载所有训练数据
    
    Args:
        base_path: 训练数据基础路径，默认为"training/"
        num_patients: 要加载的病人数量，默认为100
    
    Returns:
        x_train: 训练图像列表
        y_train: 标签图像列表
    """
    x_train = []
    y_train = []
    
    for i in range(1, num_patients + 1):
        process_patient_data(i, base_path, x_train, y_train)
    
    return x_train, y_train

def get_2D_image_count(x_train) -> int:
    tot = 0
    for i in range(len(x_train)):
        tot += x_train[i].shape[2]
    return tot

# ==================== 直接执行时的行为 ====================
if __name__ == "__main__":
    # 如果直接运行此模块，则加载数据
    x_train, y_train = load_training_data("../../../train_datas/training/")
    logger.info(f"加载完成：{len(x_train)} 个训练图像，{len(y_train)} 个标签图像")
    images_2d_count = get_2D_image_count(x_train)
    logger.info(f"2D 图像总数:{images_2d_count}")