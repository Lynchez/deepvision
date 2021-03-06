# import necessary packages
import deepvision as dv
from deepvision.object_detection import draw_bbox
import cv2
import sys

# open webcam
video = cv2.VideoCapture(sys.argv[1])

if not video.isOpened():
    print("Could not open video")
    exit()
    

# loop through frames
while video.isOpened():

    # read frame from webcam 
    status, frame = video.read()

    if not status:
        break

    # apply object detection
    bbox, label, conf = dv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

    print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    # display output
    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
video.release()
cv2.destroyAllWindows()        
