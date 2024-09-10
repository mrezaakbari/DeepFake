import sys, importlib, modules, modules.globals
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]

def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        if not all(hasattr(module, method) for method in FRAME_PROCESSORS_INTERFACE):
            sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
    return module

def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        FRAME_PROCESSORS_MODULES = [load_frame_processor_module(fp) for fp in frame_processors]
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state and frame_processor not in frame_processors:
            module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(module)
            modules.globals.frame_processors.append(frame_processor)
        elif not state and frame_processor in modules.globals.frame_processors:
            try:
                module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.remove(module)
                modules.globals.frame_processors.remove(frame_processor)
            except Exception:
                pass  # Optionally handle specific exceptions

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = [executor.submit(process_frames, source_path, [path], progress) for path in temp_frame_paths]
        for future in futures:
            future.result()

def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    with tqdm(total=len(frame_paths), desc='Processing', unit='frame', dynamic_ncols=True) as progress:
        progress.set_postfix({'execution_providers': modules.globals.execution_providers, 
                              'execution_threads': modules.globals.execution_threads, 
                              'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)
