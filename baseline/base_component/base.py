import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D, 
    UpSampling2D, 
    concatenate, 
    BatchNormalization, 
    Activation, 
    Add, 
    multiply,
    Dropout
)

def conv_block(inputs, filters, dropout_rate=0.1):
    """
    基础卷积块：Conv2D -> BN -> ReLU -> Conv2D -> BN -> ReLU -> Dropout
    """
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    return x

# -------------------------------
# U-Net++
# -------------------------------
def build_unet_plus_plus_custom(input_shape=(256, 256, 1), num_classes=4):
    """
    手动实现 U-Net++ 架构 (针对 256x256 输入)
    """
    def standard_unit(input_tensor, stage, col, filters):
        x = Conv2D(filters, (3, 3), padding='same', name=f'conv{stage}_{col}_1')(input_tensor)
        x = BatchNormalization(name=f'bn{stage}_{col}_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', name=f'conv{stage}_{col}_2')(x)
        x = BatchNormalization(name=f'bn{stage}_{col}_2')(x)
        x = Activation('relu')(x)
        return x

    nb_filter = [32, 64, 128, 256, 512]
    img_input = Input(shape=input_shape)

    # 主骨干 - 第一列 (Encoder)
    X0_0 = standard_unit(img_input, 0, 0, nb_filter[0])
    X1_0 = standard_unit(MaxPooling2D((2, 2))(X0_0), 1, 0, nb_filter[1])
    X2_0 = standard_unit(MaxPooling2D((2, 2))(X1_0), 2, 0, nb_filter[2])
    X3_0 = standard_unit(MaxPooling2D((2, 2))(X2_0), 3, 0, nb_filter[3])
    X4_0 = standard_unit(MaxPooling2D((2, 2))(X3_0), 4, 0, nb_filter[4])

    # 嵌套层 (Dense Skip Connections)
    X0_1 = standard_unit(concatenate([X0_0, UpSampling2D((2, 2))(X1_0)]), 0, 1, nb_filter[0])
    X1_1 = standard_unit(concatenate([X1_0, UpSampling2D((2, 2))(X2_0)]), 1, 1, nb_filter[1])
    X2_1 = standard_unit(concatenate([X2_0, UpSampling2D((2, 2))(X3_0)]), 2, 1, nb_filter[2])
    X3_1 = standard_unit(concatenate([X3_0, UpSampling2D((2, 2))(X4_0)]), 3, 1, nb_filter[3])

    X0_2 = standard_unit(concatenate([X0_0, X0_1, UpSampling2D((2, 2))(X1_1)]), 0, 2, nb_filter[0])
    X1_2 = standard_unit(concatenate([X1_0, X1_1, UpSampling2D((2, 2))(X2_1)]), 1, 2, nb_filter[1])
    X2_2 = standard_unit(concatenate([X2_0, X2_1, UpSampling2D((2, 2))(X3_1)]), 2, 2, nb_filter[2])

    X0_3 = standard_unit(concatenate([X0_0, X0_1, X0_2, UpSampling2D((2, 2))(X1_2)]), 0, 3, nb_filter[0])
    X1_3 = standard_unit(concatenate([X1_0, X1_1, X1_2, UpSampling2D((2, 2))(X2_2)]), 1, 3, nb_filter[1])

    X0_4 = standard_unit(concatenate([X0_0, X0_1, X0_2, X0_3, UpSampling2D((2, 2))(X1_3)]), 0, 4, nb_filter[0])

    # 输出层
    output = Conv2D(num_classes, (1, 1), activation='softmax', name='output')(X0_4)
    model = Model(inputs=img_input, outputs=output, name='UnetPlusPlus')
    return model


def attention_gate(x, g, inter_shape):
    """
    修正后的 Attention Gate (Additive Attention)
    x: 来自编码器的特征 (Skip connection), 例如 32x32
    g: 来自解码器的门控信号 (Gating signal), 例如 32x32
    inter_shape: 中间线性变换的通道数
    """
    # 1. 将跳跃连接 x 映射到中间空间
    theta_x = Conv2D(inter_shape, (1, 1), padding='same')(x)
    
    # 2. 将门控信号 g 映射到中间空间
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    
    # 3. 相加融合并过激活函数
    f = Add()([theta_x, phi_g])
    f = Activation('relu')(f)
    
    # 4. 计算注意力权重 (0~1 之间)
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    sigmoid_psi = Activation('sigmoid')(psi_f)
    
    # 5. 将权重应用到原始跳跃连接 x 上
    return multiply([x, sigmoid_psi])

def build_attention_unet_custom(input_shape=(256, 256, 1), num_classes=4):
    """手动构建修正后的 Attention U-Net"""
    inputs = Input(shape=input_shape)
    
    # --- Encoder ---
    c1 = conv_block(inputs, 32)   # 256x256
    p1 = MaxPooling2D((2, 2))(c1) # 128x128
    
    c2 = conv_block(p1, 64)       # 128x128
    p2 = MaxPooling2D((2, 2))(c2) # 64x64
    
    c3 = conv_block(p2, 128)      # 64x64
    p3 = MaxPooling2D((2, 2))(c3) # 32x32
    
    c4 = conv_block(p3, 256)      # 32x32
    p4 = MaxPooling2D((2, 2))(c4) # 16x16
    
    # --- Bottleneck ---
    c5 = conv_block(p4, 512)      # 16x16
    
    # --- Decoder with Attention ---
    # Block 6: 16x16 -> 32x32
    g6 = UpSampling2D((2, 2))(c5) 
    a6 = attention_gate(x=c4, g=g6, inter_shape=256) # c4和g6现在都是32x32
    m6 = concatenate([g6, a6])
    c6 = conv_block(m6, 256)
    
    # Block 7: 32x32 -> 64x64
    g7 = UpSampling2D((2, 2))(c6)
    a7 = attention_gate(x=c3, g=g7, inter_shape=128) # c3和g7现在都是64x64
    m7 = concatenate([g7, a7])
    c7 = conv_block(m7, 128)
    
    # Block 8: 64x64 -> 128x128
    g8 = UpSampling2D((2, 2))(c7)
    a8 = attention_gate(x=c2, g=g8, inter_shape=64)  # c2和g8现在都是128x128
    m8 = concatenate([g8, a8])
    c8 = conv_block(m8, 64)
    
    # Block 9: 128x128 -> 256x256
    g9 = UpSampling2D((2, 2))(c8)
    a9 = attention_gate(x=c1, g=g9, inter_shape=32)  # c1和g9现在都是256x256
    m9 = concatenate([g9, a9])
    c9 = conv_block(m9, 32)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    return Model(inputs, outputs, name="Attention_UNet")