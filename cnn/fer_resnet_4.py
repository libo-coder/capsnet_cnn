import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow import keras
import numpy as np

from CapsNet_vs_CNN.dataset import fer2013_4

# Training parameters
batch_size = 128    # orig paper trained all networks with batch_size=128
epochs = 4
data_augmentation = True

# 减去像素均值提高准确性
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 12

# 从提供的模型参数n计算深度
depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv2' % depth


##################
#   学习率的选取   #
##################
def lr_schedule(epoch):

    lr = 0.01
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu',
                 batch_normalization=True, conv_first=True):

    conv = keras.layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=7):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = keras.layers.Input(shape=input_shape)
    # 在分成2个路径之前，v2在输入时执行带有BN-ReLU的Conv2D
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # 实例化剩余单元的堆栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:          # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:          # first layer but not first stage
                    strides = 2             # downsample

            # 瓶颈剩余单元
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)

            if res_block == 0:
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=7)(x)
    y = keras.layers.Flatten()(x)

    outputs = keras.layers.Dense(fer2013_4.CLASSES,
                                 activation='softmax',
                                 kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Trian History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    model = resnet_v2(input_shape=fer2013_4.IMAGE_SHAPE, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models/fer_resnet_4(2)')
    model_name = 'fer_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                 monitor='val_acc',
                                                 verbose=1,
                                                 save_best_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

    # keras.callbacks.ReduceLROnPlateau 当标准评估停止提升时，降低学习速率。
    # 当学习停止时，模型总是会受益于降低 2-10 倍的学习速率。
    # 这个回调函数监测一个数据并且当这个数据在一定「有耐心」的训练轮之后还没有进步， 那么学习速率就会被降低。
    # factor: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
    # patience: 没有进步的训练轮数，在这之后训练速率会被降低。
    # min_lr: 学习速率的下边界。
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.5),
                                                   cooldown=0,
                                                   patience=5,
                                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Fit the model on the batches generated by datagen.flow().
    train_history = model.fit_generator(fer2013_4.get_train_generator_for_cnn(batch_size),
                                        steps_per_epoch=fer2013_4.TRAIN_SIZE / batch_size,
                                        validation_data=fer2013_4.get_validation_data_for_cnn(),
                                        epochs=epochs,
                                        verbose=1,
                                        workers=1,
                                        callbacks=callbacks)

    show_train_history(train_history, 'acc', 'val_acc')

    show_train_history(train_history, 'loss', 'val_loss')

    # Score trained model.
    x_test, y_test = fer2013_4.get_test_data_for_cnn()
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])          # 损失值 loss
    print('Test accuracy:', scores[1])      # 准确率 acc
