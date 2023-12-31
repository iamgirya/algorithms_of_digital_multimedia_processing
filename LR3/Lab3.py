import cv2
import numpy as np


def task():
    # 1 - строим матрицу
    # задаём значения и картинку
    img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

    standard_deviation = 100
    kernel_size = 5
    imgBlurByMe1 = MyGaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow(str(kernel_size)+'x'+str(kernel_size) +
               ' and ' + str(standard_deviation), imgBlurByMe1)

    # 4 - другие параметры
    standard_deviation = 50
    kernel_size = 11
    imgBlurByMe2 = MyGaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow(str(kernel_size)+'x'+str(kernel_size) +
               ' and ' + str(standard_deviation), imgBlurByMe2)

    # 5 - блюр от либы
    imgBlurByCV2 = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), standard_deviation)

    cv2.imshow('img', img)
    cv2.imshow('By CV2', imgBlurByCV2)
    cv2.waitKey(0)


def MyGaussianBlur(img, kernel_size, standard_deviation):
    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size+1) // 2

    # Строим матрицу свёртки
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)

    print(kernel)

    # 2 -Нормализуем
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i, j]
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum
    print(kernel)

    # 3 - юзаем матрицу
    # Проходимся по внутренним пикселям
    imgBlurByMe = svertka(img, kernel)

    return imgBlurByMe


def svertka(img, kernel):
    kernel_size = len(kernel)
    imgBlurByMe = img.copy()
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlurByMe.shape[0]-x_start):
        for j in range(y_start, imgBlurByMe.shape[1]-y_start):

            # операция свёртки
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k, j + l] * kernel[k +
                                                      (kernel_size//2), l + (kernel_size//2)]
            imgBlurByMe[i, j] = val

    return imgBlurByMe


def gauss(x, y, omega, a, b):
    omegaIn2 = 2 * omega ** 2
    m1 = 1/(np.pi * omegaIn2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2)/omegaIn2)
    return m1*m2


task()
