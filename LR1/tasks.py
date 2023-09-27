import requests
import cv2
import numpy as np


def task2():
    img1 = cv2.imread(r'.\source\img1.jpg', cv2.IMREAD_GRAYSCALE)  # ЧБ
    img2 = cv2.imread(r'.\source\img2.png',
                      cv2.IMREAD_UNCHANGED)  # без изменений
    # с учетом  глубины цвета
    img3 = cv2.imread(r'.\source\img3.bmp', cv2.IMREAD_ANYDEPTH)
    cv2.namedWindow('lol', cv2.WINDOW_NORMAL)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.namedWindow('kek', cv2.WINDOW_NORMAL)
    cv2.imshow('lol', img1)
    cv2.imshow('test', img2)
    cv2.imshow('kek', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task3():
    cap = cv2.VideoCapture(r'.\source\video.mp4', cv2.CAP_ANY)
    new_width = 640
    new_height = 480
    while True:
        ret, frame = cap.read()
        if not ret:
            exit()

        frame = cv2.resize(frame, (new_width, new_height))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Video', gray_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()


def task4():
    video = cv2.VideoCapture(r'.\source\video.mp4', cv2.CAP_ANY)
    ok, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("./Output/output3.mp4", fourcc, 25, (w, h))

    while (True):
        ok, vid = video.read()
        if (not ok):
            break
        cv2.imshow('Video', vid)
        video_writer.write(vid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


def task5():

    img1 = cv2.imread(r'.\source\img2.png')
    img2 = cv2.imread(r'.\source\img2.png')

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.namedWindow('test_hsv', cv2.WINDOW_NORMAL)

    cv2.imshow('test', img1)

    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    cv2.imshow('test_hsv', hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task6():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        cross_image = np.zeros((height, width, 3), dtype=np.uint8)

        vertical_line_width = 60
        vertical_line_height = 300

        center_x, center_y = width//2, height//2
        radius = 120
        pentagon_points = []
        for i in range(5):
            x = center_x + int(radius * np.cos(2 * np.pi * i / 5))
            y = center_y + int(radius * np.sin(2 * np.pi * i / 5))
            pentagon_points.append((x, y))

        # Рисуем пентаграмму на изображении
        for i in range(5):
            cv2.line(cross_image, pentagon_points[i], pentagon_points[(
                i + 2) % 5], (0, 0, 255), 2)

        cv2.circle(cross_image, (center_x, center_y), 120,
                   (0, 0, 255), 2)  # -1 для заполнения круга цветом

        horizontal_line_width = 250
        horizontal_line_height = 55

        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

        cv2.imshow("Red Cross", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task7():
    video = cv2.VideoCapture(0)

    ok, vid = video.read()

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        "./Output/output_webcam7.mp4", fourcc, 25, (w, h))

    while (True):
        ok, vid = video.read()

        cv2.imshow('Video', vid)
        video_writer.write(vid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


def task8():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        cross_image = np.zeros((height, width, 3), dtype=np.uint8)

        vertical_line_width = 5
        vertical_line_height = 300
        rect_start_v = (width // 2 - vertical_line_width // 2,
                        height // 2 - vertical_line_height // 2)
        rect_end_v = (width // 2 + vertical_line_width // 2,
                      height // 2 + vertical_line_height // 2)

        horizontal_line_width = 250
        horizontal_line_height = 5

        rect_start_h = (width // 2 - horizontal_line_width // 2,
                        height // 2 - horizontal_line_height // 2)
        rect_end_h = (width // 2 + horizontal_line_width // 2,
                      height // 2 + horizontal_line_height // 2)

        central_pixel_color = frame[height // 2, width // 2]

        color_distances = [
            np.linalg.norm(central_pixel_color - np.array([0, 0, 255])),
            np.linalg.norm(central_pixel_color - np.array([0, 255, 0])),
            np.linalg.norm(central_pixel_color - np.array([255, 0, 0]))
        ]

        closest_color_index = np.argmin(color_distances)

        if closest_color_index == 0:
            cv2.rectangle(cross_image, rect_start_h,
                          rect_end_h, (0, 0, 255), -1)
        elif closest_color_index == 1:
            cv2.rectangle(cross_image, rect_start_h,
                          rect_end_h, (0, 255, 0), -1)
        else:
            cv2.rectangle(cross_image, rect_start_h,
                          rect_end_h, (255, 0, 0), -1)

        if closest_color_index == 0:
            cv2.rectangle(cross_image, rect_start_v,
                          rect_end_v, (0, 0, 255), -1)
        elif closest_color_index == 1:
            cv2.rectangle(cross_image, rect_start_v,
                          rect_end_v, (0, 255, 0), -1)
        else:
            cv2.rectangle(cross_image, rect_start_v,
                          rect_end_v, (255, 0, 0), -1)

        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)

        cv2.imshow("Colored Cross", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def task9():
    url = "http://192.168.204.1:8080/shot.jpg"

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        img = cv2.resize(img, (640, 480))
        cv2.imshow("Android_cam", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


task6()
