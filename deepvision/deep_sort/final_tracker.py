import numpy as np
import cv2
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
tracking = True

# Deep SORT
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def Track(boxes, classes, confidence, frame):
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
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    if tracking:
        tracker.predict()
        tracker.update(detections)
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        cv2.putText(frame, classes[0] + " - ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 0), 1)
    """
    for det in detections:
        bbox = det.to_tlbr()
        score = "%.2f" % round(det.confidence * 100, 2) + "%"
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        if len(classes) > 0:
            cls = det.cls
            cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                        1.5e-3 * frame.shape[0], (0, 255, 0), 1)
    """
