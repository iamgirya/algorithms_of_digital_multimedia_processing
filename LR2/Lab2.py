import cv2
import numpy as np


lower_red = np.array([0, 110, 0])
upper_red = np.array([15, 200, 255])


def task2():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_only = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Red', red_only)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task3():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        in_range = cv2.inRange(hsv, lower_red, upper_red)
        kernel = np.ones((5, 5), np.uint8)

        image_opening = cv2.morphologyEx(in_range, cv2.MORPH_OPEN, kernel)
        image_closing = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Open", image_opening)
        cv2.imshow("Close", image_closing)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task4():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        in_range = cv2.inRange(hsv, lower_red, upper_red)
        red_only = cv2.bitwise_and(frame, frame, mask=in_range)
        cv2.imshow('Red', red_only)

        moments = cv2.moments(in_range)
        area = moments['m00']
        print(area)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task5():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        in_range = cv2.inRange(hsv, lower_red, upper_red)

        moments = cv2.moments(in_range)
        area = moments['m00']
        if area > 0:
            width = height = int(np.sqrt(area))
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])

            cv2.rectangle(
                frame,
                (c_x - (width // 32), c_y - (height // 32)),
                (c_x + (width // 32), c_y + (height // 32)),
                (0, 0, 0),
                2
            )

        cv2.imshow('Rectangle', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


task5()


def erode(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded


def dilate(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated
