import shutil, cv2, numpy as np, insightface, modules.globals
from tqdm import tqdm
from typing import Any, List, Dict
from pathlib import Path
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import (
    get_temp_directory_path, create_temp, extract_frames,
    clean_temp, get_temp_frame_paths
)

# Global face analysis object
FACE_ANALYSER = None

def get_face_analyser() -> Any:
    """Initializes and returns the face analyzer."""
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    """Gets a single face from the given frame."""
    faces = get_face_analyser().get(frame)
    return min(faces, key=lambda x: x.bbox[0]) if faces else None


def get_many_faces(frame: Frame) -> List[Any]:
    """Gets multiple faces from the given frame."""
    return get_face_analyser().get(frame) if frame is not None else []


def has_valid_map() -> bool:
    """Checks if there is a valid source-target map."""
    return any("source" in m and "target" in m for m in modules.globals.souce_target_map)


def default_source_face() -> Any:
    """Returns the default source face from the map."""
    for map in modules.globals.souce_target_map:
        if "source" in map:
            return map['source']['face']
    return None


def simplify_maps() -> None:
    """Simplifies the source-target face mappings to embeddings."""
    centroids, faces = [], []
    
    for map in modules.globals.souce_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])
    
    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}


def add_blank_map() -> None:
    """Adds a blank map entry with a unique ID."""
    max_id = max((m['id'] for m in modules.globals.souce_target_map), default=-1)
    modules.globals.souce_target_map.append({'id': max_id + 1})


def get_unique_faces_from_target_image() -> None:
    """Extracts unique faces from the target image and maps them."""
    try:
        modules.globals.souce_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        faces = get_many_faces(target_frame)

        for i, face in enumerate(faces):
            x_min, y_min, x_max, y_max = face['bbox']
            cropped_face = target_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            modules.globals.souce_target_map.append({
                'id': i,
                'target': {'cv2': cropped_face, 'face': face}
            })
    except Exception:
        pass


def get_unique_faces_from_target_video() -> None:
    """Extracts unique faces from a target video and maps them."""
    try:
        modules.globals.souce_target_map = []
        frame_face_embeddings, face_embeddings = [], []

        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        for i, temp_frame_path in enumerate(tqdm(temp_frame_paths, desc="Extracting face embeddings")):
            temp_frame = cv2.imread(temp_frame_path)
            faces = get_many_faces(temp_frame)
            
            face_embeddings.extend([face.normed_embedding for face in faces])
            frame_face_embeddings.append({'frame': i, 'faces': faces, 'location': temp_frame_path})

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        for i, _ in enumerate(centroids):
            modules.globals.souce_target_map.append({'id': i, 'target_faces_in_frame': []})

            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroid-{i}"):
                mapped_faces = [face for face in frame['faces'] if face['target_centroid'] == i]
                if mapped_faces:
                    modules.globals.souce_target_map[i]['target_faces_in_frame'].append(
                        {'frame': frame['frame'], 'faces': mapped_faces, 'location': frame['location']}
                    )

        default_target_face()
    except Exception:
        pass


def default_target_face() -> None:
    """Selects the best face as the default target face in the map."""
    for map in modules.globals.souce_target_map:
        best_face, best_frame = None, None
        for frame in map['target_faces_in_frame']:
            if frame['faces']:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        if best_face:
            x_min, y_min, x_max, y_max = best_face['bbox']
            target_frame = cv2.imread(best_frame['location'])
            map['target'] = {
                'cv2': target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                'face': best_face
            }


def dump_faces(centroids: Any, frame_face_embeddings: List[Dict]) -> None:
    """Saves cropped face images in directories based on their cluster centroids."""
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i, _ in enumerate(centroids):
        cluster_dir = Path(temp_directory_path) / str(i)
        if cluster_dir.exists():
            shutil.rmtree(cluster_dir)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/{i}"):
            temp_frame = cv2.imread(frame['location'])

            for j, face in enumerate(frame['faces']):
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']
                    cropped_face = temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    if cropped_face.size > 0:
                        cv2.imwrite(cluster_dir / f"{frame['frame']}_{j}.png", cropped_face)
