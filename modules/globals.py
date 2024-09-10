import os
from typing import List, Dict, Any

# Define root and workflow directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

# File type mappings
file_types = [
    ('Image', ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')),
    ('Video', ('*.mp4', '*.mkv'))
]

# Source to target mappings
source_target_map: List[Dict[str, Any]] = []
simple_map: Dict[str, str] = {}

# Paths for source, target, and output
source_path: str = None
target_path: str = None
output_path: str = None

# Frame processing options
frame_processors: List[str] = []

# Global flags and configurations
keep_fps: bool = None
keep_audio: bool = None
keep_frames: bool = None
many_faces: bool = None
map_faces: bool = None
color_correction: bool = None  # Toggle for color correction
nsfw_filter: bool = None

# Video encoding settings
video_encoder: str = None
video_quality: str = None

# Live stream options
live_mirror: bool = None
live_resizable: bool = None

# System and execution settings
max_memory: int = None
execution_providers: List[str] = []
execution_threads: int = None
headless: bool = None

# Logging level
log_level: str = 'error'

# UI and camera settings
fp_ui: Dict[str, bool] = {}
camera_input_combobox: Any = None
webcam_preview_running: bool = False
