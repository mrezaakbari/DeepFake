import glob, mimetypes, os, platform, shutil, ssl, subprocess, urllib, modules.globals
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'


if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context

def clean_temp(target_path: str) -> None:
    """Clean up the temporary directory after processing."""
    temp_directory_path = get_temp_directory_path(target_path)
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)

def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    # """Download files if they do not exist in the directory."""
    os.makedirs(download_directory_path, exist_ok=True)
    for url in urls:
        file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(file_path):
            with tqdm(total=int(urllib.request.urlopen(url).headers.get('Content-Length', 0)), desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, file_path, reporthook=lambda c, b, t: progress.update(b))  # type: ignore

def create_temp(target_path: str) -> None:
    """Create a temporary directory."""
    Path(get_temp_directory_path(target_path)).mkdir(parents=True, exist_ok=True)

def create_video(target_path: str, fps: float = 30.0) -> None:
    """Create a video from extracted frames."""
    run_ffmpeg(['-r', str(fps), '-i', os.path.join(get_temp_directory_path(target_path), '%04d.png'), '-c:v', modules.globals.video_encoder, '-crf', str(modules.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', get_temp_output_path(target_path)])

def detect_fps(target_path: str) -> float:
    """Detect the frames per second (FPS) of a video."""
    try:
        numerator, denominator = map(int, subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]).decode().strip().split('/'))
        return numerator / denominator
    except Exception:
        return 30.0

def extract_frames(target_path: str) -> None:
    """Extract frames from a video."""
    run_ffmpeg(['-i', target_path, '-pix_fmt', 'rgb24', os.path.join(get_temp_directory_path(target_path), '%04d.png')])

def get_temp_directory_path(target_path: str) -> str:
    """Get the path to the temporary directory for a video."""
    return os.path.join(os.path.dirname(target_path), TEMP_DIRECTORY, os.path.splitext(os.path.basename(target_path))[0])

def get_temp_frame_paths(target_path: str) -> List[str]:
    """Get paths to temporary frames."""
    return glob.glob(os.path.join(glob.escape(get_temp_directory_path(target_path)), '*.png'))

def get_temp_output_path(target_path: str) -> str:
    """Get the path to the temporary output video file."""
    return os.path.join(get_temp_directory_path(target_path), TEMP_FILE)

def has_image_extension(image_path: str) -> bool:
    """Check if the file has an image extension."""
    return image_path.lower().endswith(('png', 'jpg', 'jpeg'))

def is_image(image_path: str) -> bool:
    """Check if the file is an image based on MIME type."""
    return bool(image_path and os.path.isfile(image_path) and mimetypes.guess_type(image_path)[0].startswith('image/'))

def is_video(video_path: str) -> bool:
    """Check if the file is a video based on MIME type."""
    return bool(video_path and os.path.isfile(video_path) and mimetypes.guess_type(video_path)[0].startswith('video/'))

def move_temp(target_path: str, output_path: str) -> None:
    """Move the temporary video file to the final output location."""
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)

def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    """Normalize the output path for saving the result."""
    if os.path.isdir(output_path):
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        return os.path.join(output_path, f"{source_name}-{target_name}{target_extension}")
    return output_path

def resolve_relative_path(path: str) -> str:
    """Resolve a relative path to an absolute one."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def restore_audio(target_path: str, output_path: str) -> None:
    """Restore audio from the original video."""
    if not run_ffmpeg(['-i', get_temp_output_path(target_path), '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path]):
        move_temp(target_path, output_path)

def run_ffmpeg(args: List[str]) -> bool:
    """Run an FFmpeg command with given arguments."""
    try:
        subprocess.check_output(['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', modules.globals.log_level] + args, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False
