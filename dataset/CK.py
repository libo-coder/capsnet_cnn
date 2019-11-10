import functools
import os
import cv2
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (48, 48, 1)
TRAIN_SIZE = 686
VALIDATION_SIZE = 236
TEST_SIZE = 59
CLASSES = 7

# rotation range的作用是用户指定旋转角度范围，其参数只需指定一个整数即可，但并不是固定以这个角度进行旋转,而是在[0, 指定角度]范围内进行随机角度旋转
# width_shift_range & height_shift_range 分别是水平位置平移和上下位置平移，其参数可以是[0, 1]的浮点数，也可以大于1，
# 其最大平移距离为图片长或宽的尺寸乘以参数，同样平移距离并不固定为最大平移距离，平移距离在 [0, 最大平移距离] 区间内。
# horizontal_flip的作用是随机对图片执行水平翻转操作，意味着不一定对所有图片都会执行水平翻转，每次生成均是随机选取图片进行翻转。
# vertical_flip是作用是对图片执行上下翻转操作，和horizontal_flip一样，每次生成均是随机选取图片进行翻转
RANDOM_ROTATION_CONFIG = {
    'rotation_range': 30,       # Random rotations from -30 deg to 30 deg
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
}

# RANDOM_ROTATION_CONFIG = {
#     'rotation_range': 0.,       # Random rotations from -30 deg to 30 deg
#     'width_shift_range': 0.,
#     'height_shift_range': 0.,
#     'horizontal_flip': False,
#     'vertical_flip': False,     # Doesn't make sense in FER2013
# }

@functools.lru_cache()
def _load_data():

    trainPath = '/Users/libo/Documents/Deep_Learning/datasets/CK+48'

    train_label = []

    train_total = []

    ##################
    #    加载 train   #
    ##################
    for root, dirs, files in os.walk(trainPath):
        for filename in (x for x in files if x.endswith(('.jpg', '.tiff', '.png'))):
            filepath1 = os.path.join(root, filename)
            object_class = filepath1.split('\\')[-1]    # 情感标签
            if object_class == '0':
                train_label.append(0)
            elif object_class == '1':
                train_label.append(1)
            elif object_class == '2':
                train_label.append(2)
            elif object_class == '3':
                train_label.append(3)
            elif object_class == '4':
                train_label.append(4)
            elif object_class == '5':
                train_label.append(5)
            else:
                train_label.append(6)
            image = np.array(Image.open(filepath1))
            # ddd = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            train_total.append(image)
    train_total2 = np.array(train_total)
    print('训练图片总维度', train_total2.shape)

    X = np.array(train_total2)
    y = np.array(train_label)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    x_train = x_train.reshape(-1, 48, 48, 1).astype('float32')
    x_train /= 255
    print(x_train.shape[0], 'train samples')
    print('X_train:', x_train.shape)
    y_train = keras.utils.to_categorical(y_train.astype('float32'))
    print('y_train.shape:', y_train.shape)

    x_validation = x_validation.reshape(-1, 48, 48, 1).astype('float32')
    x_validation /= 255
    print(x_validation.shape[0], 'validation samples')
    print('x_validation:', x_validation.shape)
    y_validation = keras.utils.to_categorical(y_validation.astype('float32'))
    print('y_validation.shape:', y_validation.shape)

    x_test = x_test.reshape(-1, 48, 48, 1).astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    print('x_test:', x_test.shape)
    y_test = keras.utils.to_categorical(y_test.astype('float32'))
    print('y_test.shape:', y_test.shape)

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

# _load_data()


def get_train_generator_for_cnn(batch_size):
    (x_train, y_train), (_, _), (_, _) = _load_data()
    train_datagen = keras.preprocessing.image.ImageDataGenerator(**RANDOM_ROTATION_CONFIG)
    # generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
    while 1:
        x_batch, y_batch = generator.next()
        yield [x_batch, y_batch]
        # print("1")

def get_train_generator_for_capsnet(batch_size):
    for (x_batch, y_batch) in get_train_generator_for_cnn(batch_size):
        yield ([x_batch, y_batch], [y_batch, x_batch])
        # print("libo")


def get_validation_data_for_cnn():
    (_, _), (x_validation, y_validation), (_, _) = _load_data()
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(**RANDOM_ROTATION_CONFIG)
    generator = validation_datagen.flow(x_validation, y_validation, batch_size=1)

    x_validation = np.empty_like(x_validation)
    y_validation = np.empty_like(y_validation)

    for i, (x_batch, y_batch) in enumerate(generator):
        if i >= VALIDATION_SIZE:
            break
        x_validation[i:(i+1)] = x_batch[:]
        y_validation[i:(i+1)] = y_batch[:]
        # print('libo2')

    return [x_validation, y_validation]


def get_validation_data_for_capsnet():
    x_validation, y_validation = get_validation_data_for_cnn()
    return [[x_validation, y_validation], [y_validation, x_validation]]


def get_test_data_for_cnn(rotation=0.0):
    (_, _), (_, _), (x_test, y_test) = _load_data()
    x_test = np.array([keras.preprocessing.image.apply_affine_transform(image, theta=rotation) for image in x_test])
    return (x_test, y_test)


# def get_test_data_for_capsnet(rotation=0.0):
#     x_test, y_test = get_test_data_for_cnn(rotation)
#     return [[x_test, y_test], [y_test, x_test]]


def get_test_data_for_capsnet():
    x_test, y_test = get_test_data_for_cnn()
    return [[x_test, y_test], [y_test, x_test]]

