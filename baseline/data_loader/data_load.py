from typing import Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import os

# 这里假定项目里存在 lib.py 提供以下两个函数
# from lib import load_training_data, get_2D_image_count
# 如果你的 lib 在包内不同位置，请在导入处调整
try:
    from .lib import load_training_data, get_2D_image_count  # type: ignore
except Exception as e:
    # 延迟错误：仅在真正调用加载时抛出更可读的错误
    load_training_data = None  # type: ignore
    get_2D_image_count = None  # type: ignore


def find_project_root(start_path: Path, marker: str = "train_datas") -> Path:
    """从 start_path 向上查找包含 marker 文件夹的目录"""
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"找不到项目根目录，缺少 {marker}")


def convert_3d_to_2d(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 3D 图像数据转换为 2D 切片（假定 shape 为 H x W x D）"""
    if len(x_train) != len(y_train):
        raise ValueError("x_train 与 y_train 数量不匹配")
    total_slices = sum(int(img.shape[2]) for img in x_train)
    if total_slices == 0:
        raise ValueError("没有检测到任何切片 (total_slices == 0)")

    # 假定每张切片为 256x256；若不是，请调整或在调用前归一化/reshape
    H, W = int(x_train[0].shape[0]), int(x_train[0].shape[1])
    x_2d = np.zeros((total_slices, H, W), dtype=x_train[0].dtype)
    y_2d = np.zeros((total_slices, H, W), dtype=y_train[0].dtype)

    index = 0
    for i in range(len(x_train)):
        num_slices = int(x_train[i].shape[2])
        for j in range(num_slices):
            x_2d[index] = x_train[i][:, :, j]
            y_2d[index] = y_train[i][:, :, j]
            index += 1

    return x_2d, y_2d


def split_and_shuffle(
    x_data: np.ndarray,
    y_data: np.ndarray,
    train_size: int = 1200,
    val_size: int = 200,
    seed: int = 10
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """划分并随机打乱数据集。若数据不足，自动调整大小以适配总样本数。"""
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("x_data 与 y_data 第一维长度不一致")

    total = x_data.shape[0]
    # 自动调整：如果请求的大小超过总样本数，则按比例或最小保证划分
    if train_size + val_size >= total:
        logger.warning(
            f"请求的 train_size+val_size ({train_size}+{val_size}) >= 总样本数 {total}，将自动调整划分比例"
        )
        # 保证至少每集合 5% 左右（或至少 1 个），但保留一个较大训练集
        train_size = max(1, int(total * 0.7))
        val_size = max(1, int(total * 0.15))
        logger.info(f"调整为 train={train_size}, val={val_size}, test={total-train_size-val_size}")

    x_train = x_data[:train_size]
    y_train = y_data[:train_size]

    x_val = x_data[train_size:train_size + val_size]
    y_val = y_data[train_size:train_size + val_size]

    x_test = x_data[train_size + val_size:]
    y_test = y_data[train_size + val_size:]

    rng = np.random.RandomState(seed)

    train_idx = rng.permutation(x_train.shape[0])
    val_idx = rng.permutation(x_val.shape[0]) if x_val.shape[0] > 0 else np.array([], dtype=int)
    test_idx = rng.permutation(x_test.shape[0]) if x_test.shape[0] > 0 else np.array([], dtype=int)

    x_train, y_train = x_train[train_idx], y_train[train_idx]
    if x_val.shape[0] > 0:
        x_val, y_val = x_val[val_idx], y_val[val_idx]
    if x_test.shape[0] > 0:
        x_test, y_test = x_test[test_idx], y_test[test_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def add_channel_dim(arr: np.ndarray) -> np.ndarray:
    """在末尾添加一个单通道维度，比如 (N,H,W) -> (N,H,W,1)"""
    return np.expand_dims(arr, axis=3)


def visualize_sample(x_data: np.ndarray, y_data: np.ndarray, index: int = 4, save_path: Optional[str] = None) -> None:
    """可视化样本图像和标签，并自动创建保存路径（如果提供 save_path）"""
    if index < 0 or index >= x_data.shape[0]:
        raise IndexError(f"index 超出范围: 0 <= index < {x_data.shape[0]}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(x_data[index, :, :, 0], cmap='gray', interpolation='none')
    plt.title(f"输入图像 (样本 {index})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(y_data[index, :, :, 0], interpolation='none')
    plt.title(f"标签图像 (样本 {index})")
    plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"可视化结果已保存至: {save_path}")
    plt.show()


def create_and_save_npy(
    target_folder: str,
    train_folder: Optional[str] = None,
    *,
    train_size: int = 1200,
    val_size: int = 200,
    seed: int = 10,
    sample_index: int = 4,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    主流程函数：
    - target_folder: 要保存 .npy 文件的目标文件夹（若不存在会自动创建）
    - train_folder: (可选) 训练数据文件夹路径（如果 None，则会尝试通过 find_project_root 检索）
    - train_size, val_size, seed: 划分参数
    - sample_index: 可视化样本索引
    - visualize: 是否在完成后保存并显示示例图像

    返回值: dict 包含数组对象与保存的路径
    """
    # 检查外部依赖导入
    if load_training_data is None or get_2D_image_count is None:
        raise ImportError("无法导入 `load_training_data` 或 `get_2D_image_count`。请确认 lib.py 在 import 路径中并导出这两个函数。")

    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"目标保存目录: {target_path.resolve()}")

    # 确定训练数据文件夹
    if train_folder:
        TRAIN_FOLDER = Path(train_folder)
    else:
        current_dir = Path(__file__).parent
        project_root = find_project_root(current_dir, marker="train_datas")
        TRAIN_FOLDER = project_root / "train_datas" / "training"

    if not TRAIN_FOLDER.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {TRAIN_FOLDER}")

    # 加载数据（用户提供的函数）
    x_train_3d, y_train_3d = load_training_data(TRAIN_FOLDER)
    logger.info(f"加载完成：{len(x_train_3d)} 个训练病例 (3D)，{len(y_train_3d)} 个标签病例(3D)")

    # 可选：输出 2D 总数
    try:
        images_2d_count = get_2D_image_count(x_train_3d)
        logger.info(f"2D 图像总数 (估计): {images_2d_count}")
    except Exception:
        logger.debug("get_2D_image_count 调用失败或返回异常，继续执行转换流程。")

    # 转换为 2D 切片
    x_2d, y_2d = convert_3d_to_2d(x_train_3d, y_train_3d)
    logger.info(f"转换完成：{x_2d.shape[0]} 个 2D 切片")

    # 划分并打乱
    (x_2d_train, y_2d_train), (x_2d_val, y_2d_val), (x_2d_test, y_2d_test) = \
        split_and_shuffle(x_2d, y_2d, train_size=train_size, val_size=val_size, seed=seed)
    logger.info(f"划分完成：训练 {x_2d_train.shape[0]}, 验证 {x_2d_val.shape[0]}, 测试 {x_2d_test.shape[0]}")

    # 添加通道维度
    x_2d_train = add_channel_dim(x_2d_train)
    y_2d_train = add_channel_dim(y_2d_train)

    x_2d_val = add_channel_dim(x_2d_val)
    y_2d_val = add_channel_dim(y_2d_val)

    x_2d_test = add_channel_dim(x_2d_test)
    y_2d_test = add_channel_dim(y_2d_test)

    logger.info(f"数据形状 - 训练: {x_2d_train.shape}, 验证: {x_2d_val.shape}, 测试: {x_2d_test.shape}")

    # 保存 6 个文件
    save_paths = {}
    save_paths['x_2d_train'] = str(target_path / "x_2d_train.npy")
    save_paths['y_2d_train'] = str(target_path / "y_2d_train.npy")
    save_paths['x_2d_val'] = str(target_path / "x_2d_val.npy")
    save_paths['y_2d_val'] = str(target_path / "y_2d_val.npy")
    save_paths['x_2d_test'] = str(target_path / "x_2d_test.npy")
    save_paths['y_2d_test'] = str(target_path / "y_2d_test.npy")

    np.save(save_paths['x_2d_train'], x_2d_train)
    np.save(save_paths['y_2d_train'], y_2d_train)
    np.save(save_paths['x_2d_val'], x_2d_val)
    np.save(save_paths['y_2d_val'], y_2d_val)
    np.save(save_paths['x_2d_test'], x_2d_test)
    np.save(save_paths['y_2d_test'], y_2d_test)

    logger.info("已保存 6 个 .npy 文件到目标目录。")

    # 可视化(可以解开注释，如果您需要的话)
    vis_path = None
    if visualize and x_2d_train.shape[0] > 0:
        vis_path = str(target_path / f"acdc_sample_{sample_index}.png")
        try:
            logger.info("准备做可视化展示: ")
            # visualize_sample(x_2d_train, y_2d_train, index=sample_index, save_path=vis_path)
        except Exception as e:
            logger.warning(f"可视化失败: {e}")

    result = {
        "x_2d_train": x_2d_train,
        "y_2d_train": y_2d_train,
        "x_2d_val": x_2d_val,
        "y_2d_val": y_2d_val,
        "x_2d_test": x_2d_test,
        "y_2d_test": y_2d_test,
        "save_paths": save_paths,
        "visualization": vis_path
    }
    return result