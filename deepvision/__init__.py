__version__ = "0.0.36"

from .face_detection import detect_face
from .object_detection import detect_common_objects, TFlite, YOLO
from .gender_detection import detect_gender
from .utils import get_frames, animate
from .object_tracker import Track
from .deep_sort import *
from .generate_detections import *