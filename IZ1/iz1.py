import cv2
import numpy as np


def iz_part1(file, tracker_type):
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL().create()  # norm-sbit
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF().create()  # norm-lost
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting.create()  # hren'
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD.create()  # norm-lagaet
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow().create()  # hren'
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN().create()  # hren'
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE.create()  # norm-lost
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT().create()  # norm-lost
    if tracker_type == "NANO":
        params = cv2.TrackerNano.Params()
        params.backbone = 'nanotrack_backbone_sim.onnx'
        params.neckhead = 'nanotrack_head_sim.onnx'
        tracker = cv2.TrackerNano().create(parameters=params)  # norm-perfect
    if tracker_type == "DASIAMRPN":
        tracker = cv2.TrackerDaSiamRPN().create()  # norm-perfect

    # Read video
    video = cv2.VideoCapture(r"IZ1\imgs\\" + file)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"IZ1\output\video2_" + file +
                             tracker_type + ".mp4", fourcc, 90, (w, h))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        return

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video")
        return

    # Define an initial bounding box
    bbox = (250, 153, 156, 198)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)
        writer.write(frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    writer.release()


class MeanShiftTracker(object):
    def __init__(self, curWindowRoi, imgBGR):
        self.updateCurrentWindow(curWindowRoi)
        self.updateHistograms(imgBGR)

        self.term_criteria = (cv2.TERM_CRITERIA_EPS |
                              cv2.TERM_CRITERIA_COUNT, 10, 1)

    def updateCurrentWindow(self,  curWindowRoi):

        self.curWindow = curWindowRoi

    def updateHistograms(self, imgBGR):
        '''
          update the histogram and rois according to the current object in the current image

        '''

        self.bgrObjectRoi = imgBGR[self.curWindow[1]: self.curWindow[1] + self.curWindow[3],
                                   self.curWindow[0]: self.curWindow[0] + self.curWindow[2]]
        self.hsvObjectRoi = cv2.cvtColor(self.bgrObjectRoi, cv2.COLOR_BGR2HSV)

        # get the mask for calculating histogram and also remove some noise
        self.mask = cv2.inRange(self.hsvObjectRoi, np.array(
            (0., 50., 50.)), np.array((180, 255., 255.)))

        # use 180 bins for each H value, and normalize the histogram to lie b/w [0, 255]
        self.histObjectRoi = cv2.calcHist(
            [self.hsvObjectRoi], [0], self.mask, [180], [0, 180])
        cv2.normalize(self.histObjectRoi, self.histObjectRoi,
                      0, 255, cv2.NORM_MINMAX)

    def getBackProjectedImage(self, imgBGR):
        '''
           convert the current BGR image, imgBGR, to HSV color space 
           and return the backProjectedImg
        '''
        # print("[info] getBackprjectImage calls", imgBGR.shape)
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

        # obtained the back projected image using the histogram obtained earlier

        backProjectedImg = cv2.calcBackProject(
            [imgHSV], [0], self.histObjectRoi, [0, 180], 1)

        self.backProjectedImg = backProjectedImg

        return backProjectedImg.copy()

    def computeNewWindow(self, imgBGR):
        '''
            Track the window enclosing the object of interest using meanShift function of openCV for the 
            current frame imgBGR
        '''

        self.getBackProjectedImage(imgBGR)

        _, curWindow = cv2.meanShift(
            self.backProjectedImg, self.curWindow, self.term_criteria)

        self.updateCurrentWindow(curWindow)

    def getCurWindow(self):

        return self.curWindow


def iz_part2():
    cap = cv2.VideoCapture(r"IZ1\imgs\video1.mp4")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"IZ1\output\video2_" +
                             "MeanShift.mp4", fourcc, 90, (w, h))
    ret, frame = cap.read()
    x, y, w, h = cv2.selectROI(frame, False)
    track_window = (x, y, w, h)
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (True):
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

            cv2.putText(frame, "MeanShift Tracker", (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            cv2.imshow('img2', img2)
            writer.write(img2)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    writer.release()
    cv2.destroyAllWindows()


# TODO видосы добавить
tracker_types = ['KCF', 'CSRT', 'DASIAMRPN']
files = ['video1.mp4', '2.mov', '3.mov', '4.mov', '5.mov']
for tracker_type in tracker_types:
    for file in files:
        iz_part1(file, tracker_type)


# iz_part2()
