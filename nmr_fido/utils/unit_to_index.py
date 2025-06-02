from __future__ import annotations
import numpy as np
from nmr_fido import NMRData
from typing import TypeVar

NMRArrayType = TypeVar("NMRArrayType", bound=np.ndarray)

def _convert_to_index(
    data: NMRArrayType,
    value,
    npoints: int,
    default: int,
    dim: int = -1,
) -> int:
    """
    Convert a string like "5.5 ppm" or "1234 pts" into an integer index.
    Negative values count from the end of the array.
    """
    if value is None or not isinstance(data, NMRData):
        return default

    if isinstance(value, int):
        idx = value if value >= 0 else npoints + value
        return int(np.clip(idx, 0, npoints - 1))

    if isinstance(value, str):
        cleaned = value.strip().lower().replace(" ", "")
        number_part = (
            cleaned.replace("ppm", "")
            .replace("hz", "")
            .replace("pts", "")
            .replace("%", "")
        )
        
        try:
            number = float(number_part)
        except ValueError:
            raise ValueError(f"Could not parse value: {value}")
        
        # Get correct scale for the specified dimension
        scale = data.axes[dim]["scale"]

        if "ppm" in value.lower() or "hz" in value.lower():
            idx = np.argmin(np.abs(scale - number))
            return int(idx)
        
        elif "pts" in value.lower():
            idx = number if number >= 0 else npoints + number
            return int(np.clip(idx, 0, npoints - 1))
        
        elif "%" in value.lower():
            idx = (number / 100.0) * npoints
            idx = idx if idx >= 0 else npoints + idx
            return int(np.clip(idx, 0, npoints - 1))
        
        else:
            # Assume pts if no unit
            idx = number if number >= 0 else npoints + number
            return int(np.clip(idx, 0, npoints - 1))

    raise ValueError(f"Invalid start/end value: {value}")