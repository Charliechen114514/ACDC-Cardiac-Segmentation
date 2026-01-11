from baseline.data_loader import load_data, check_npy_files_exist
from baseline.train_core import train_model, train_model_light_uent, train_model_unetpp,train_model_attention_unet
from baseline.evaluate import evaluate, evaluate_unetlight, evaluate_unetpp, evaluate_attention_unet
from loguru import logger
from pathlib import Path

# ------------- Settings -------------
DATA_FOLDER = "train_datas/training/"
BASE_RESULT_DIR_RAW = "baseline/result"
EPOCHES = 70
# ------------- Settings End -------------

BASE_RESULT_DIR = Path(BASE_RESULT_DIR_RAW)
SAVE_NPY_FOLDER = BASE_RESULT_DIR / "acdc_npy"
MODEL_BASE_FOLDER = BASE_RESULT_DIR / "models"

owns_missing, _ = check_npy_files_exist(SAVE_NPY_FOLDER)
if not owns_missing:
    logger.info(f"folder {DATA_FOLDER} missing npy files or not existed! Auto Regenerates datas...")
    load_data(DATA_FOLDER, SAVE_NPY_FOLDER)
else:
    logger.info(f"√ Data has been ready!")

logger.info("*" * 60)
logger.info("Enter train model session!")

# Vgg16
model_path = train_model(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
evaluate("baseline/result", SAVE_NPY_FOLDER, model_path)

# Light Unet
model_path_light_unet = train_model_light_uent(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
evaluate_unetlight("baseline/result", SAVE_NPY_FOLDER, model_path_light_unet)

# 3) U-Net++
model_path_unetpp = train_model_unetpp(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
evaluate_unetpp("baseline/result", SAVE_NPY_FOLDER, model_path_unetpp)

# 4) Attention U-Net
model_path_attention = train_model_attention_unet(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
evaluate_attention_unet("baseline/result", SAVE_NPY_FOLDER, model_path_attention)

logger.info("All experiments finished.")
