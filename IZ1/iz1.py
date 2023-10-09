import cv2
import numpy as np


def iz(tracker_type):
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
        tracker = cv2.TrackerNano().create(parameters=params) # norm-perfect
    if tracker_type == "DASIAMRPN":
        tracker = cv2.TrackerDaSiamRPN().create()

    # Read video
    video = cv2.VideoCapture(r"IZ1\imgs\video1.mp4")

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(r"IZ1\output\video1_" +
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
    # bbox = cv2.selectROI(frame, False)

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


tracker_types = ['DASIAMRPN', 'NANO']
for tracker_type in tracker_types:
    iz(tracker_type)
