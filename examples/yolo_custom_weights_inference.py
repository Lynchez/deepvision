
# import necessary packages
import deepvision as dv
from deepvision.object_detection import YOLO
import sys
import cv2

weights = sys.argv[1]
config = sys.argv[2]
labels = sys.argv[3]

# read input image
image = cv2.imread(sys.argv[4])

yolo = YOLO(weights, config, labels)

# apply object detection
bbox, label, conf = yolo.detect_objects(image)

print(bbox, label, conf)

# draw bounding box over detected objects
yolo.draw_bbox(image, bbox, label, conf, write_conf=True)

# display output
# press any key to close window           
cv2.imshow("object_detection", image)
cv2.waitKey()

# save output
cv2.imwrite("yolo_object_detection.jpg", image)

# release resources
cv2.destroyAllWindows()

