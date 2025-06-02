import numpy as np


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