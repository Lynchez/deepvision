__version__ = "0.0.2"

from .face_detection import detect_face
from .object_detection import detect_common_objects
from .gender_detection import detect_gender
from .utils import get_frames, animate
from .object_tracker import Track