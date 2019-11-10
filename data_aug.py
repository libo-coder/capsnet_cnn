import numpy as np
import cv2
import PIL.Image
import random

PATH = 'data/data_aug/'

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def colormap(img, img_name):
    rand_b = rand() + 1
    rand_g = rand() + 1
    rand_r = rand() + 1
    H, W, C = img.shape
    new_name = img_name + '_color.png'
    dst = np.zeros((H, W, C), np.uint8)
    for i in range(H):
        for j in range(W):
            (b, g, r) = img[i, j]
            b = int(b * rand_b)
            g = int(g * rand_g)
            r = int(r * rand_r)
            if b > 255:
                b = 255
            if g > 255:
                g = 255
            if r > 255:
                r = 255
            dst[i][j] = (b, g, r)
    cv2.imwrite(PATH + new_name, dst)


def blur(img, img_name):
    img_GaussianBlur = cv2.GaussianBlur(img, (5, 5), 0)
    img_Mean = cv2.blur(img, (5, 5))
    img_Median = cv2.medianBlur(img, 3)
    img_Bilater = cv2.bilateralFilter(img, 5, 100, 100)

    new_name = img_name + '_gaussianblur.png'
    new_name1 = img_name + '_blur.png'
    new_name2 = img_name + '_medianblur.png'
    new_name3 = img_name + '_bilateralfilter.png'

    cv2.imwrite(PATH + new_name, img_GaussianBlur)
    cv2.imwrite(PATH + new_name1, img_Mean)
    cv2.imwrite(PATH + new_name2, img_Median)
    cv2.imwrite(PATH + new_name3, img_Bilater)


def noise(img, img_name):
    H, W, C = img.shape
    noise_img = np.zeros((H, W, C), np.uint8)

    for i in range(H):
        for j in range(W):
            noise_img[i, j] = img[i, j]

    for i in range(100):
        x = np.random.randint(H)
        y = np.random.randint(W)
        noise_img[x, y, :] = 255

    new_name = img_name + '_noise.png'
    cv2.imwrite(PATH + new_name, noise_img)


def main(img, img_name):
    colormap(img, img_name)
    blur(img, img_name)
    noise(img, img_name)
    print(img_name + "增强完成")


if __name__ == '__main__':
    img = cv2.imread('data/jaffe_64/HA/KA.HA2.30.png')
    img_name = 'KA.HA2.30'
    main(img, img_name)