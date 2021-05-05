import numpy as np
import cv2
from .deep_sort import *
from .generate_detections import *
from .utils import download_file

max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
tracking = True
dest_dir = os.path.expanduser('~') + os.path.sep + '.deepvision' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
colors = np.random.uniform(0, 255, size=(80, 3))

# Deep SORT
model_filename = 'mars-small128.pb'
model_path = dest_dir + os.path.sep + model_filename
cfg_url = 'https://github.com/Lynchez/deepvision/raw/master/model_data/mars-small128.pb'

if not os.path.exists(model_path):
    download_file(url=cfg_url, file_name=model_filename, dest_dir=dest_dir)

encoder = create_box_encoder(model_path, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def Track(boxes, classes, confidence, frame):

    for i, label in enumerate(classes):

        color = colors[classes.index(label)]

        cv2.rectangle(frame, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), color, 2)

        cv2.putText(frame, label, (boxes[i][0],boxes[i][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if tracking:
        features = encoder(frame, boxes)

        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                        zip(boxes, confidence, classes, features)]
    else:
        detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                        zip(boxes, confidence, classes)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    if tracking:
        tracker.predict()
        tracker.update(detections)
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                    0.5, (0, 255, 0), 2)

    return frame