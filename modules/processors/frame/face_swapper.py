import cv2, insightface, threading, modules.globals, modules.processors.frame.core
from typing import Any, List
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.cluster_analysis import find_closest_centroid

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    return True

def pre_start() -> bool:
    source_path_valid = is_image(modules.globals.source_path) and (modules.globals.map_faces or get_one_face(cv2.imread(modules.globals.source_path)))
    target_path_valid = is_image(modules.globals.target_path) or is_video(modules.globals.target_path)
    
    if not source_path_valid:
        update_status('Select an image for source path or no face detected.', NAME)
        return False
    if not target_path_valid:
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128_fp16.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=modules.globals.execution_providers)
    return FACE_SWAPPER

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    target_faces = get_many_faces(temp_frame) if modules.globals.many_faces else [get_one_face(temp_frame)]
    
    for target_face in target_faces:
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    source_face = default_source_face() if modules.globals.many_faces else None

    for map in modules.globals.souce_target_map:
        target_faces = [f for f in map['target_faces_in_frame'] if f['location'] == temp_frame_path] if is_video(modules.globals.target_path) else [map['target']['face']]
        
        if source_face is None and "source" in map:
            source_face = map['source']['face']
        
        for target_face in target_faces:
            if target_face['faces']:
                for face in target_face['faces']:
                    temp_frame = swap_face(source_face, face, temp_frame)

    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path)) if not modules.globals.map_faces else None
    
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame(source_face or process_frame_v2(temp_frame, temp_frame_path), temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
        if progress:
            progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path)) if not modules.globals.map_faces else None
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame) if source_face else process_frame_v2(target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status('Many faces enabled. Using first source image. Progressing...', NAME)
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
