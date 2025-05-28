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
    

def get_hz_scale(npoints: int, sw: float, ori: float) -> np.ndarray:
    """
    Generate an Hz frequency scale for an NMR spectrum.

    Args:
        npoints (int): Number of points in the spectrum.
        sw (float): Sweep width in Hz.
        ori (float): Origin frequency (center of the spectrum) in Hz.

    Returns:
        np.ndarray: Hz scale array, decreasing from high to low frequency.
    """
    points = np.arange(npoints)
    
    o1_Hz = ori + sw / 2 - sw / npoints
    hz_scale = o1_Hz - sw * (points / npoints - 0.5)
    
    return hz_scale



def get_ppm_scale(npoints: int, sw: float, ori: float, obs: float) -> np.ndarray:
    """
    Generate a ppm scale for an NMR spectrum.

    Args:
        npoints (int): Number of points in the spectrum.
        sw (float): Sweep width in Hz.
        ori (float): Origin frequency (center of the spectrum) in Hz.
        obs (float): Spectrometer frequency in MHz.

    Returns:
        np.ndarray: ppm scale array.
    """
    hz_scale = get_hz_scale(npoints, sw, ori)
    
    ppm_scale = hz_scale / obs
    
    return ppm_scale