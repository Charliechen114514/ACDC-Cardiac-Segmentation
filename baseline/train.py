import os
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 防 OOM / GPU 设置
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 启用按需增长，避免一次性占满显存导致 OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# -------------------------------
# 参数（可按需微调 batch_size 防止显存爆炸）
# -------------------------------
BATCH_SIZE = 4         # 如果仍 OOM，请降低到 2 或 1
EPOCHS = 50
NUM_CLASSES = 4
BACKBONE = "vgg16"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax"
INPUT_SHAPE = (256, 256, 3)   # base_model 所需的输入 (因为我们会 map 1->3)

# 保存路径（保持你原来的风格）
SAVE_BASE_PATH = "."
MODEL_SAVE_DIR = os.path.join(SAVE_BASE_PATH, "risultati100epochs")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_NAME = os.path.join(MODEL_SAVE_DIR, f"modelUnet_{EPOCHS}epochs.keras")

# -------------------------------
# 辅助：generator 使用 mmap，按需做标准化与 one-hot
# -------------------------------
def np_generator(x_npy_path, y_npy_path):
    """
    从磁盘按样本逐个读取（使用 numpy mmap 模式），返回 (x, y)：
    - x: shape (256,256,1), float32, 已经除以该切片的最大值（若 max=0 则原样不变）
    - y: shape (256,256,4), float32, one-hot 按 (label==1,2,3,0)
    保证：不把全部数据一次性加载到内存。
    """
    x_mmap = np.load(x_npy_path, mmap_mode='r')
    y_mmap = np.load(y_npy_path, mmap_mode='r')
    n_samples = x_mmap.shape[0]
    for i in range(n_samples):
        # x: (256,256,1)
        x_slice = x_mmap[i, :, :, 0].astype(np.float32)
        maxv = np.max(x_slice)
        if maxv > 0:
            x_slice = x_slice / maxv
        x_slice = np.expand_dims(x_slice, axis=-1)

        # y: one-hot 4 channels; 按原逻辑映射：ch0=(==1), ch1=(==2), ch2=(==3), ch3=(==0)
        y_slice_raw = y_mmap[i, :, :, 0]
        h, w = y_slice_raw.shape
        y_onehot = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
        y_onehot[:, :, 0] = (y_slice_raw == 1)
        y_onehot[:, :, 1] = (y_slice_raw == 2)
        y_onehot[:, :, 2] = (y_slice_raw == 3)
        y_onehot[:, :, 3] = (y_slice_raw == 0)

        yield x_slice, y_onehot

# -------------------------------
# 构建 tf.data.Dataset（懒加载）
# -------------------------------
def make_dataset(x_path, y_path, batch_size):
    output_signature = (
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, NUM_CLASSES), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: np_generator(x_path, y_path),
        output_signature=output_signature
    )
    # 不做 cache（会占内存），不做 shuffle（保持原有顺序/逻辑）
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# 路径（假设 .npy 文件位于当前工作目录）
X_TRAIN_PATH = "x_2d_train.npy"
Y_TRAIN_PATH = "y_2d_train.npy"
X_VAL_PATH = "x_2d_val.npy"
Y_VAL_PATH = "y_2d_val.npy"
X_TEST_PATH = "x_2d_test.npy"
Y_TEST_PATH = "y_2d_test.npy"

train_ds = make_dataset(X_TRAIN_PATH, Y_TRAIN_PATH, BATCH_SIZE)
val_ds = make_dataset(X_VAL_PATH, Y_VAL_PATH, BATCH_SIZE)
test_ds = make_dataset(X_TEST_PATH, Y_TEST_PATH, BATCH_SIZE)

# -------------------------------
# 构建模型（与原逻辑等价：先用 base_model 接受 3 通道，然后用 Conv2D 将 1->3）
# -------------------------------
base_model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=INPUT_SHAPE,
    classes=NUM_CLASSES,
    activation=ACTIVATION,
    encoder_weights=ENCODER_WEIGHTS
)

# 使用任意空间大小输入，但通道数为 1（数据生成器输出）
inp = Input(shape=(None, None, 1))
x = Conv2D(3, (1, 1))(inp)  # 将单通道映射到 3 通道，保持原逻辑
out = base_model(x)
modelUnet = Model(inp, out, name=base_model.name)

# -------------------------------
# 编译模型（保持原损失/指标不变）
# -------------------------------
lr = 1e-4
optim = Adam(learning_rate=lr)

dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + focal_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

modelUnet.compile(optimizer=optim, loss=total_loss, metrics=metrics)

# -------------------------------
# 训练（使用 tf.data 流式训练以避免 OOM）
# -------------------------------
history = modelUnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    # 若需要更稳健的训练监控，可以添加 callbacks（例如 ModelCheckpoint），
    # 但原始逻辑只在训练结束时保存最终模型，所以这里保持一致。
)

# -------------------------------
# 在 test 集上评估（与原逻辑保持一致：提供测试评估）
# -------------------------------
test_metrics = modelUnet.evaluate(test_ds)
print("Test metrics:", test_metrics)

# -------------------------------
# 保存模型（与原逻辑相同）
# -------------------------------
modelUnet.save(MODEL_SAVE_NAME)
print(f"Model saved to: {MODEL_SAVE_NAME}")
