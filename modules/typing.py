from typing import Any
import numpy as np
from insightface.app.common import Face

# Alias types for readability
Face = Face
Frame = np.ndarray[Any, Any]

"""
Defines alias types for face detection and frame handling.
- Face: Represents a face object from the InsightFace library.
- Frame: Represents a NumPy array used for image frames.
"""
