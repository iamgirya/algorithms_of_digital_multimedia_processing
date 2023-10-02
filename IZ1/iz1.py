import cv2
import numpy as np


def svertka(img, kernel):
    kernel_size = len(kernel)
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    matr = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr[i][j] = img[i][j]

    for i in range(x_start, len(matr)-x_start):
        for j in range(y_start, len(matr[i])-y_start):

            # операция свёртки
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k][j + l] * kernel[k +
                                                      (kernel_size//2)][l + (kernel_size//2)]
            matr[i][j] = val

    return matr


def get_angle_number(x, y):
    tg = y/x if x != 0 else 999

    if (x < 0):
        if (y < 0):
            if (tg > 2.414):
                return 0
            elif (tg < 0.414):
                return 6
            elif (tg <= 2.414):
                return 7
        else:
            if (tg < -2.414):
                return 4
            elif (tg < -0.414):
                return 5
            elif (tg >= -0.414):
                return 6
    else:
        if (y < 0):
            if (tg < -2.414):
                return 0
            elif (tg < -0.414):
                return 1
            elif (tg >= -0.414):
                return 2
        else:
            if (tg < 0.414):
                return 2
            elif (tg < 2.414):
                return 3
            elif (tg >= 2.414):
                return 4


def canny(kernel_size, standard_deviation, bound_path, operator_mode, path):
    file_name = 'img_' + path + '_ks_' + str(kernel_size) + '_sd_' + str(
        standard_deviation) + '_bp_' + str(bound_path) + '_om_' + str(operator_mode) + '.jpg'

    # make gray img
    frame = cv2.imread(r'.\IZ1\imgs\\' + path, cv2.IMREAD_GRAYSCALE)

    # blur
    img = cv2.GaussianBlur(
        frame, (kernel_size, kernel_size), standard_deviation)

    # choose operator
    if (operator_mode == 'sobel'):
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    elif (operator_mode == 'roberts'):
        Gx = [[1, 0], [0, -1]]
        Gy = [[0, 1], [-1, 0]]
    elif (operator_mode == 'kirsch'):
        G1 = [[5,  5,  5],
              [-3,  0, -3],
              [-3, -3, -3]]
        G2 = [[-3,  5,  5],
              [-3,  0, 5],
              [-3, -3, -3]]
        G3 = [[-3,  -3,  5],
              [-3,  0, 5],
              [-3, -3, 5]]
        G4 = [[-3,  -3,  -3],
              [-3,  0, 5],
              [-3, 5, 5]]
        G5 = [[-3,  -3,  -3],
              [-3,  0, -3],
              [5, 5, 5]]
        G6 = [[-3,  -3,  -3],
              [5,  0, -3],
              [5, 5, -3]]
        G7 = [[5,  -3,  -3],
              [5,  0, -3],
              [5, -3, -3]]
        G8 = [[5,  5,  -3],
              [5,  0, -3],
              [-3, -3, -3]]
        G = [G1, G2, G3, G4, G5, G6, G7, G8]
        img_G = []
        for gn in G:
            img_G.append(svertka(img, gn))

    matr_gradient = np.zeros(img.shape)
    # считаем градиент
    if (operator_mode == 'sobel' or operator_mode == 'roberts'):
        img_Gx = svertka(img, Gx)
        img_Gy = svertka(img, Gy)
        max_gradient = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                matr_gradient[i][j] = np.sqrt(
                    img_Gx[i][j] ** 2 + img_Gy[i][j] ** 2)
                if (matr_gradient[i][j] > max_gradient):
                    max_gradient = matr_gradient[i][j]
    else:
        max_gradient = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                gn_values = []
                for n in range(len(img_G)):
                    gn_values.append(img_G[n][i][j])
                matr_gradient[i][j] = np.max(gn_values)
                if (matr_gradient[i][j] > max_gradient):
                    max_gradient = matr_gradient[i][j]

    # считаем углы
    matr_angles = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_angles[i][j] = get_angle_number(img_Gx[i][j], img_Gy[i][j])

    # бордеры без фильтров
    img_border_not_filtered = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            angle = matr_angles[i][j]
            gradient = matr_gradient[i][j]
            if (i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1):
                img_border_not_filtered[i][j] = 0
            else:
                x_shift = 0
                y_shift = 0
                if (angle == 0 or angle == 4):
                    x_shift = 0
                elif (angle > 0 and angle < 4):
                    x_shift = 1
                else:
                    x_shift = -1

                if (angle == 2 or angle == 6):
                    y_shift = 0
                elif (angle > 2 and angle < 6):
                    y_shift = -1
                else:
                    y_shift = 1

                is_max = gradient >= matr_gradient[i+y_shift][j +
                                                              x_shift] and gradient >= matr_gradient[i-y_shift][j-x_shift]
                img_border_not_filtered[i][j] = 255 if is_max else 0

    # подавляем немаксимумы
    lower_bound = max_gradient/bound_path
    upper_bound = max_gradient - max_gradient/bound_path
    img_border_filtered = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradient = matr_gradient[i][j]
            if (img_border_not_filtered[i][j] == 255):
                if (gradient >= lower_bound and gradient <= upper_bound):
                    flag = False
                    for k in range(-1, 2):
                        if (flag):
                            break
                        for l in range(-1, 2):
                            if (img_border_not_filtered[i+k][j+l] == 255 and matr_gradient[i+k][j+l] >= lower_bound):
                                flag = True
                                break
                    if (flag):
                        img_border_filtered[i][j] = 255
                elif (gradient > upper_bound):
                    img_border_filtered[i][j] = 255

    cv2.imwrite(r'.\IZ1\output\\' + file_name, img_border_filtered)


kernel_sizes = [3, 7, 11]
standard_deviations = [1.4]
bound_paths = [10, 7, 4]
operators = ['sobel', 'roberts', 'kirsch']
imgs = ['test.jpg']

for k in kernel_sizes:
    for s in standard_deviations:
        for b in bound_paths:
            for o in operators:
                for i in imgs:
                    canny(k, s, b, o, i)
