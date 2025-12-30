# import os
# import fnmatch
# import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt

# # -------------------------------
# # 函数定义
# # -------------------------------

# def find(pattern, path):
#     """
#     遍历指定路径 path，查找匹配 pattern 的文件。
#     返回匹配文件的完整路径列表。
#     """
#     result = []
#     for root, dirs, files in os.walk(path):
#         for name in files:
#             if fnmatch.fnmatch(name, pattern):
#                 result.append(os.path.join(root, name))
#     return result


# def padding_image(img):
#     """
#     对输入的三维图像 img 进行填充，使其大小为 256x256。
#     填充方式为在右下角填充零。
#     """
#     dim1, dim2, dim3 = img.shape
#     new_image = np.zeros(shape=(256, 256, dim3))
#     new_image[0:dim1, 0:dim2, 0:dim3] = img
#     return new_image


# def str_fun(i):
#     """
#     将整数 i 转换为三位字符串。
#     1 -> '001', 12 -> '012', 123 -> '123'
#     """
#     if i <= 9:
#         return "00" + str(i)
#     elif i <= 99:
#         return "0" + str(i)
#     else:
#         return str(i)


# # -------------------------------
# # 数据加载与预处理
# # -------------------------------

# x_train = []  # 存储原始图像
# y_train = []  # 存储标签图像

# for i in range(1, 101):
#     # 构建病人路径
#     path_iniz = "../train_datas/training/"
#     patient = "patient" + str_fun(i)
#     path_iniz += patient

#     # 查找该病人下所有帧文件
#     patient_pattern = patient + "_frame*"
#     result = find(patient_pattern, path_iniz)

#     result_other = []

#     # 先处理 frame01 文件
#     for name in result:
#         if fnmatch.fnmatch(name, "*frame01"):
#             if fnmatch.fnmatch(name, "*_gt.nii.gz"):
#                 # 标签文件
#                 img = nib.load(name)
#                 if np.max(img.shape) < 257:
#                     y_train.append(padding_image(img.get_fdata()))
#             else:
#                 # 原始图像
#                 img = nib.load(name)
#                 if np.max(img.shape) < 257:
#                     x_train.append(padding_image(img.get_fdata()))
#         else:
#             result_other.append(name)

#     # 处理其余帧
#     for name in result_other:
#         if fnmatch.fnmatch(name, "*_gt.nii.gz"):
#             img = nib.load(name)
#             if np.max(img.shape) < 257:
#                 y_train.append(padding_image(img.get_fdata()))
#         else:
#             img = nib.load(name)
#             if np.max(img.shape) < 257:
#                 x_train.append(padding_image(img.get_fdata()))


# # -------------------------------
# # 将 3D 图像展开为 2D 切片
# # -------------------------------

# # 统计总切片数量
# tot = sum(img.shape[2] for img in x_train)

# # 创建 2D 图像数组
# x_2d = np.zeros(shape=(tot, 256, 256))
# y_2d = np.zeros(shape=(tot, 256, 256))

# # 填充 2D 图像
# index = 0
# for i in range(len(x_train)):
#     for ii in range(x_train[i].shape[2]):
#         x_2d[index + ii, :, :] = x_train[i][:, :, ii]
#         y_2d[index + ii, :, :] = y_train[i][:, :, ii]
#     index += x_train[i].shape[2]

# # -------------------------------
# # 数据集划分
# # -------------------------------

# x_2d_train = x_2d[0:1200, :, :]
# y_2d_train = y_2d[0:1200, :, :]

# x_2d_val = x_2d[1200:1400, :, :]
# y_2d_val = y_2d[1200:1400, :, :]

# x_2d_test = x_2d[1400:1812, :, :]
# y_2d_test = y_2d[1400:1812, :, :]

# # -------------------------------
# # 打乱训练集、验证集、测试集
# # -------------------------------

# np.random.seed(10)

# index = np.random.permutation(1200)
# x_2d_train = x_2d_train[index, :, :]
# y_2d_train = y_2d_train[index, :, :]

# index = np.random.permutation(200)
# x_2d_val = x_2d_val[index, :, :]
# y_2d_val = y_2d_val[index, :, :]

# index = np.random.permutation(412)
# x_2d_test = x_2d_test[index, :, :]
# y_2d_test = y_2d_test[index, :, :]

# # -------------------------------
# # 扩展维度以适配深度学习输入
# # -------------------------------

# x_2d_train = np.expand_dims(x_2d_train, axis=3)
# y_2d_train = np.expand_dims(y_2d_train, axis=3)

# x_2d_val = np.expand_dims(x_2d_val, axis=3)
# y_2d_val = np.expand_dims(y_2d_val, axis=3)

# x_2d_test = np.expand_dims(x_2d_test, axis=3)
# y_2d_test = np.expand_dims(y_2d_test, axis=3)

# # -------------------------------
# # 保存为 .npy 文件
# # -------------------------------

# np.save("x_2d_train.npy", x_2d_train)
# np.save("y_2d_train.npy", y_2d_train)

# np.save("x_2d_val.npy", x_2d_val)
# np.save("y_2d_val.npy", y_2d_val)

# np.save("x_2d_test.npy", x_2d_test)
# np.save("y_2d_test.npy", y_2d_test)

# # -------------------------------
# # 加载训练集并统计标签分布
# # -------------------------------

# x_2d_train = np.load("x_2d_train.npy")
# y_2d_train = np.load("y_2d_train.npy")

# print("标签为0的像素数量:", np.sum(y_2d_train[0, :, :, 0] == 0))
# print("标签为1的像素数量:", np.sum(y_2d_train[0, :, :, 0] == 1))
# print("标签为2的像素数量:", np.sum(y_2d_train[0, :, :, 0] == 2))
# print("标签为3的像素数量:", np.sum(y_2d_train[0, :, :, 0] == 3))

# # -------------------------------
# # 可视化示例
# # -------------------------------

# n = 4  # 显示第 n 张图像

# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(x_2d_train[n, :, :, 0], cmap='gray', interpolation='none')
# plt.subplot(1, 2, 2)
# plt.imshow(y_2d_train[n, :, :, 0])
# plt.savefig("acdc_prova.png")
# plt.show()

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

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

# -------------------------------
# 数据加载与划分
# -------------------------------

data_path = "../train_datas/training/"
# 获取所有 patient 文件夹名
patient_ids = sorted([d for d in os.listdir(data_path) if d.startswith('patient')])

# 严格按病人划分数据集，防止数据泄露 (Data Leakage)
train_ids, test_val_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

def process_patient_list(p_list):
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
# 执行与保存
# -------------------------------

print(f"Total patients: {len(patient_ids)}")
print("Processing training data...")
x_train, y_train = process_patient_list(train_ids)

print("Processing validation data...")
x_val, y_val = process_patient_list(val_ids)

print("Processing test data...")
x_test, y_test = process_patient_list(test_ids)

# 保存
np.save("x_2d_train.npy", x_train.astype(np.float32))
np.save("y_2d_train.npy", y_train.astype(np.float32))
np.save("x_2d_val.npy", x_val.astype(np.float32))
np.save("y_2d_val.npy", y_val.astype(np.float32))
np.save("x_2d_test.npy", x_test.astype(np.float32))
np.save("y_2d_test.npy", y_test.astype(np.float32))

print("-" * 30)
print(f"Train slices: {x_train.shape[0]}")
print(f"Val slices:   {x_val.shape[0]}")
print(f"Test slices:  {x_test.shape[0]}")
print("All .npy files saved successfully.")