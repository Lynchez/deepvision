
# import necessary packages
import deepvision as dv
from deepvision.object_detection import YOLO
import cv2
import sys

weights = sys.argv[1]
config = sys.argv[2]
labels = sys.argv[3]

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
yolo = YOLO(weights, config, labels)

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply object detection
    bbox, label, conf = yolo.detect_objects(frame)

    print(bbox, label, conf)

    # draw bounding box over detected objects
    yolo.draw_bbox(frame, bbox, label, conf, write_conf=True)

    # display output
    cv2.imshow("Real-time object detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        
