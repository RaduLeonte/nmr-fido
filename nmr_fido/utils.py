from __future__ import annotations
import numpy as np


def _convert_to_index(
    data: 'NMRData',
    value,
    npoints: int,
    default: int,
    dim: int = -1,
) -> int:
    """
    Convert a string like "5.5 ppm" or "1234 pts" into an integer index.
    Negative values count from the end of the array.
    """
    if value is None:
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
        scale = data.scales[dim]

        if "ppm" in value.lower() or "hz" in value.lower():
            idx = np.argmin(np.abs(scale - number))
            return idx
        
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


def get_ppm_scale(npoints: int, sw: float, ori: float, obs: float) -> np.ndarray:
    points = np.arange(npoints)
    
    # Calculate the frequency (Hz) of the first point on the spectrum
    # Formula: origin + (sweep_width / 2) - (sweep_width / npoints)
    # This adjusts for the fact that the first point is slightly below the upper edge
    o1_Hz = ori + sw / 2 - sw / npoints
    
    # Generate Hz scale: symmetric around origin, decreasing from high to low frequency
    # This matches how FFT output is ordered in most NMR datasets (left = high freq)
    hz_scale = o1_Hz - sw * (points / npoints - 0.5)
    
    
    # Calculate ppm scale
    ppm_scale = hz_scale / obs


    return ppm_scale