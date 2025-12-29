from data_loader import create_and_save_npy
from loguru import logger
from baseline_train import train_main
from evaluate import save_fast_result, evaluate_model
from pathlib import Path

def pretty_print_out(out: dict):
    logger.info("========== ACDC NPY 保存结果 ==========")
    
    # 打印文件路径部分
    logger.info("文件保存路径:")
    for key, path in out.get("save_paths", {}).items():
        logger.info(f"    {key}: {path}")
    
    # 打印可视化路径
    vis = out.get("visualization", None)
    if vis:
        logger.info(f"可视化样本: {vis}")
    else:
        logger.info("可视化样本: 未生成")
    
    # 打印数组形状
    logger.info("数据数组形状:")
    for arr_key in ["x_2d_train", "y_2d_train", "x_2d_val", "y_2d_val", "x_2d_test", "y_2d_test"]:
        arr = out.get(arr_key)
        if arr is not None:
            logger.info(f"    {arr_key}: {arr.shape}")
    logger.info("=======================================")


DATA_PATH = "result/acdc_npy/"

# 定义目标保存路径
save_dir = Path(DATA_PATH)
save_dir.mkdir(parents=True, exist_ok=True)  # 文件夹不存在就创建

# 检查六个文件是否都存在
expected_files = [
    "x_2d_train.npy", "y_2d_train.npy",
    "x_2d_val.npy", "y_2d_val.npy",
    "x_2d_test.npy", "y_2d_test.npy"
]

if all((save_dir / f).exists() for f in expected_files):
    logger.info("NPY 数据已存在，跳过生成。")
    out = {
        "save_paths": {f: str(save_dir / f) for f in expected_files},
        "visualization": None,
    }
else:
    out = create_and_save_npy(str(save_dir), train_folder=None, train_size=1200, val_size=200, visualize=True)

pretty_print_out(out)

# 训练模型
model_path = train_main(str(save_dir), "result/model_result/", 1)

save_fast_result(str(save_dir), "result/evaluation/result", model_path)

evaluate_model(model_path, DATA_PATH)