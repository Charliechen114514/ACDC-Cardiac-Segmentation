from baseline.data_loader import load_data, check_npy_files_exist
from baseline.train_core import (
    train_model,
    train_model_light_uent,
    train_model_unetpp,
    train_model_attention_unet,
)
from baseline.evaluate import (
    evaluate,
    evaluate_unetlight,
    evaluate_unetpp,
    evaluate_attention_unet,
)
from baseline.log_helpers import RedirectStdoutStderrToFile
from loguru import logger
from pathlib import Path
from datetime import datetime

# ------------- Settings -------------
DATA_FOLDER = "train_datas/training/"
BASE_RESULT_DIR_RAW = "baseline/result"
EPOCHES = 60
# ------------- Settings End -------------

BASE_RESULT_DIR = Path(BASE_RESULT_DIR_RAW)
SAVE_NPY_FOLDER = BASE_RESULT_DIR / "acdc_npy"
MODEL_BASE_FOLDER = BASE_RESULT_DIR / "models"

logger_file_name = f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
log_dir = BASE_RESULT_DIR / "log"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / logger_file_name

with RedirectStdoutStderrToFile(log_path=log_path):
    logger.info(f"Logger will be redirected to: {log_path}")

    owns_missing, _ = check_npy_files_exist(SAVE_NPY_FOLDER)
    if not owns_missing:
        logger.info(
            f"folder {DATA_FOLDER} missing npy files or not existed! "
            f"Auto Regenerates datas..."
        )
        load_data(DATA_FOLDER, SAVE_NPY_FOLDER)
    else:
        logger.info("√ Data has been ready!")

    logger.info("*" * 60)
    logger.info("Enter train model session!")

    # 1) VGG16
    model_path = train_model(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
    evaluate(BASE_RESULT_DIR_RAW, SAVE_NPY_FOLDER, model_path)

    # 2) Light U-Net
    model_path_light_unet = train_model_light_uent(
        SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES
    )
    evaluate_unetlight(
        BASE_RESULT_DIR_RAW, SAVE_NPY_FOLDER, model_path_light_unet
    )

    # 3) U-Net++
    model_path_unetpp = train_model_unetpp(
        SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES
    )
    evaluate_unetpp(
        BASE_RESULT_DIR_RAW, SAVE_NPY_FOLDER, model_path_unetpp
    )

    # 4) Attention U-Net
    model_path_attention = train_model_attention_unet(
        SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES
    )
    evaluate_attention_unet(
        BASE_RESULT_DIR_RAW, SAVE_NPY_FOLDER, model_path_attention
    )

    logger.info("All experiments finished.")
