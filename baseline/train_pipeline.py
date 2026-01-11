from baseline.data_loader import load_data, check_npy_files_exist
from baseline.train_core import train_model, train_model_light_uent, train_model_unetpp,train_model_attention_unet
from baseline.evaluate import evaluate, evaluate_unetlight, evaluate_unetpp, evaluate_attention_unet
from loguru import logger

# ------------- Settings -------------
DATA_FOLDER = "train_datas/training/"
SAVE_NPY_FOLDER = "baseline/result/acdc_npy"
MODEL_BASE_FOLDER = "baseline/result/models/"
EPOCHES = 1
# ------------- Settings End -------------

owns_missing, _ = check_npy_files_exist(SAVE_NPY_FOLDER)
if not owns_missing:
    logger.info(f"folder {DATA_FOLDER} missing npy files or not existed! Auto Regenerates datas...")
    load_data(DATA_FOLDER, SAVE_NPY_FOLDER)
else:
    logger.info(f"√ Data has been ready!")

logger.info("*" * 60)
logger.info("Enter train model session!")

# # Vgg16
# model_path = train_model(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
# evaluate("./result", SAVE_NPY_FOLDER, model_path)

# # Light Unet
# model_path_light_unet = train_model_light_uent(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
# evaluate_unetlight("./result", SAVE_NPY_FOLDER, model_path_light_unet)

# 3) U-Net++
# model_path_unetpp = train_model_unetpp(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
# evaluate_unetpp("./result", SAVE_NPY_FOLDER, model_path_unetpp)

# 4) Attention U-Net
model_path_attention = train_model_attention_unet(SAVE_NPY_FOLDER, MODEL_BASE_FOLDER, EPOCHES)
evaluate_attention_unet("./result", SAVE_NPY_FOLDER, model_path_attention)

logger.info("All experiments finished.")
