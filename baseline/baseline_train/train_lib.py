#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm
from loguru import logger
from pathlib import Path
from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence
import gc

# 设置环境变量（保留与原脚本相同的行为）
os.environ['LD_LIBRARY_PATH'] = '/home/charliechen/miniconda3/envs/final_project/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# 设置 TensorFlow GPU 内存按需增长（不限制）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU memory growth enabled")
    except RuntimeError as e:
        logger.info(e)


class NpyDataGenerator(Sequence):
    """
    从 npy 文件按 batch 读取（使用 mmap），在 __getitem__ 中进行标准化与 one-hot。
    这样不会把整个数据集一次性载入内存，适用于大数据集 / OOM 情况。
    """
    def __init__(self, x_path, y_path, batch_size=2, shuffle=True, num_classes=4):
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.num_classes = int(num_classes)

        # 使用 mmap_mode='r'，不把整个文件读入内存
        self.x_mm = np.load(x_path, mmap_mode='r')
        self.y_mm = np.load(y_path, mmap_mode='r')

        # dataset info
        self.n_samples = self.x_mm.shape[0]
        # record input channels to construct model later
        self.input_channels = self.x_mm.shape[-1]

        self.indices = np.arange(self.n_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        # compute batch indices
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start:end]

        # load batch into memory (small)
        X = self.x_mm[batch_indices].astype(np.float32)  # shape: (b,h,w,c)
        y_raw = self.y_mm[batch_indices]  # keep dtype as original (likely int)

        # per-sample standardization (与原脚本行为一致：按样本最大值归一化)
        # 假设像素值在 channel 0
        if X.ndim == 4:
            for i in range(X.shape[0]):
                # 防止全部为 0 导致除零
                max_val = np.max(X[i, :, :, 0])
                if max_val > 0:
                    X[i, :, :, 0] = X[i, :, :, 0] / max_val

        # 动态生成 one-hot（batch 内）
        b, h, w = y_raw.shape[0], y_raw.shape[1], y_raw.shape[2]
        y_onehot = np.zeros((b, h, w, self.num_classes), dtype=np.float32)

        # 按你原来的标签映射：1->class0, 2->class1, 3->class2, 0->class3
        # （保留原实现的类别顺序）
        for i in range(b):
            # y_raw assumed shape (b,h,w,1) or (b,h,w)
            ay = y_raw[i]
            if ay.ndim == 3 and ay.shape[-1] == 1:
                ay = ay[..., 0]
            y_onehot[i, :, :, 0] = (ay == 1)
            y_onehot[i, :, :, 1] = (ay == 2)
            y_onehot[i, :, :, 2] = (ay == 3)
            y_onehot[i, :, :, 3] = (ay == 0)

        return X, y_onehot

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class UNetTrainer:
    """
    使用 mmap + batch 内 one-hot 的内存友好版训练器。
    保持原来训练流程、损失、backbone 等设置不变（尽量兼容原脚本行为）。
    """
    def __init__(self, data_path="./"):
        # data_path should be directory ending with '/'
        if not data_path.endswith(os.sep):
            data_path = data_path + os.sep
        self.data_path = data_path

    def check_gpu(self):
        logger.info(f"TensorFlow: {tf.__version__}")
        logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU: {gpu_devices}")

        if gpu_devices:
            logger.info("\n✅ GPU 可用！")
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.matmul(a, a)
            logger.info("GPU 测试完成")
        else:
            logger.info("\n⚠️ 未检测到 GPU")

    def _get_npy_paths_and_shape(self):
        """只返回 npy 路径与一些基本信息（不加载全部到内存）"""
        x_train_path = os.path.join(self.data_path, "x_2d_train.npy")
        y_train_path = os.path.join(self.data_path, "y_2d_train.npy")
        x_val_path = os.path.join(self.data_path, "x_2d_val.npy")
        y_val_path = os.path.join(self.data_path, "y_2d_val.npy")

        # 用 mmap 打开以获取 shape/info，但保持为 memmap 对象小开销
        x_train_mm = np.load(x_train_path, mmap_mode='r')
        x_val_mm = np.load(x_val_path, mmap_mode='r')
        # 只读取必要信息
        logger.info(f"Train shape (mmap): {x_train_mm.shape}")
        logger.info(f"Val shape   (mmap): {x_val_mm.shape}")

        # 清理 memmap 变量（generator 里会重新 load）
        del x_train_mm, x_val_mm
        gc.collect()

        return x_train_path, y_train_path, x_val_path, y_val_path

    def build_model(self, input_channels=1, num_classes=4):
        """构建 U-Net 模型（保持原始实现）"""
        logger.info("Building U-Net model...")

        backbone = "vgg16"
        input_shape = (256, 256, 3)
        encoder_weights = "imagenet"
        activation = "softmax"

        # segmentation_models Unet (输入给 base_model 是 3-channel)
        base_model = sm.Unet(
            backbone_name=backbone,
            input_shape=input_shape,
            classes=num_classes,
            activation=activation,
            encoder_weights=encoder_weights
        )

        # 封装：输入为任意 HxW x input_channels，通过 1x1 conv 扩展到 3 channels 给 base_model
        inp = Input(shape=(None, None, input_channels))
        l1 = Conv2D(3, (1, 1))(inp)
        out = base_model(l1)

        model = Model(inp, out, name=base_model.name)

        return model

    def compile_model(self, model, learning_rate=0.0001):
        logger.info("Compiling model...")

        optim = Adam(learning_rate)

        dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 0.5]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5)
        ]

        model.compile(optim, total_loss, metrics)

        return model

    def train_model(self, model, train_gen, val_gen, epochs=100):
        logger.info(f"Starting training for {epochs} epochs...")

        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

        callbacks = [
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # 注意：workers 越大越消耗系统内存，默认保守使用 2 或 0
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
            workers=2,
            use_multiprocessing=False
        )

        return history

    def save_model(self, model, epochs, output_path="./"):
        model_path = os.path.join(output_path, f"modelUnet_{epochs}epochs.keras")
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

        return model_path

    def run(self, save_mod_path, epochs=10, batch_size=2, shuffle_train=True, shuffle_val=False):
        """
        执行训练流程（内存友好）：
        - 不一次性加载所有数据，而是使用 NpyDataGenerator 从 .npy mmap 中按 batch 读取
        """
        target_path = Path(save_mod_path)
        target_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"目标保存目录: {target_path.resolve()}")

        # 检查 GPU
        self.check_gpu()

        # 仅获取 npy 路径（不加载全部到内存）
        x_train_path, y_train_path, x_val_path, y_val_path = self._get_npy_paths_and_shape()

        logger.info("\n创建数据生成器 (mmap + batch 内 one-hot)...")
        train_gen = NpyDataGenerator(x_train_path, y_train_path, batch_size=batch_size, shuffle=shuffle_train, num_classes=4)
        val_gen = NpyDataGenerator(x_val_path, y_val_path, batch_size=batch_size, shuffle=shuffle_val, num_classes=4)

        logger.info(f"Train batches: {len(train_gen)}")
        logger.info(f"Val batches: {len(val_gen)}")

        # 构建模型，使用 generator 的 input_channels（更稳健）
        model = self.build_model(input_channels=train_gen.input_channels, num_classes=4)

        # 编译模型
        model = self.compile_model(model, learning_rate=0.0001)

        # 打印模型摘要
        logger.info("\nModel Summary:")
        model.summary()

        # 训练模型
        history = self.train_model(model, train_gen, val_gen, epochs=epochs)

        # 保存模型
        model_path = self.save_model(model, epochs, output_path=save_mod_path)

        # 强制垃圾回收
        del train_gen, val_gen, model, history
        gc.collect()

        logger.info("\n✅ 训练完成！")
        return model_path


def train_main(data_path, save_mod_path, epochs=10, batch_size=2):
    trainer = UNetTrainer(data_path)
    return trainer.run(save_mod_path, epochs=epochs, batch_size=batch_size)

