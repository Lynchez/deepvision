import cvlib as cv
import cv2
from deepvision.deep_sort.final_tracker import Track

# open webcam
webcam = cv2.VideoCapture(0)

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    frame = cv2.resize(frame,(416,416), interpolation=cv2.INTER_AREA)

    # apply object detection
    boxes, classes, confidence = cv.detect_common_objects(frame, confidence=0.25, model='yolov4-tiny')
    frame = Track(boxes, classes, confidence, frame)

    # display output
    cv2.imshow("Real-time object detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()