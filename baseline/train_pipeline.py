from data_loader import load_data, check_npy_files_exist
from train_core import train_model
from evaluate import evaluate
from loguru import logger

# ------------- Settings -------------
DATA_FOLDER = "../train_datas/training/"
SAVE_NPY_FOLDER = "result/acdc_npy"
MODEL_BASE_FOLDER = "result/models/"
EPOCHES = 30
# ------------- Settings End -------------

owns_missing, _ = check_npy_files_exist(SAVE_NPY_FOLDER)
if not owns_missing:
    logger.info(f"folder {DATA_FOLDER} missing npy files or not existed! Auto Regenerates datas...")
    load_data(DATA_FOLDER, SAVE_NPY_FOLDER)
else:
    logger.info(f"√ Data has been ready!")

logger.info("*" * 60)
logger.info("Enter train model session!")

model_path = train_model(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)

evaluate("./result", SAVE_NPY_FOLDER, model_path)
