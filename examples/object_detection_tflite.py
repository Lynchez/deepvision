import cv2
from deepvision import TFlite

# open webcam
webcam = cv2.VideoCapture(0)

tflite = TFlite("/Users/nurettin/Documents/Python/test.tflite", "coconames.txt")

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    frame = tflite.detect_objects(frame)
    # display output
    cv2.imshow("Real-time object detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources

webcam.release()
cv2.destroyAllWindows()