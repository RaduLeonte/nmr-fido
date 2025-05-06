import numpy as np
from nmr_fido.nmrdata import NMRData

def _convert_to_index(
    data: NMRData,
    value,
    npoints: int,
    default: int,
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

        if "ppm" in value.lower():
            # Find nearest ppm
            scale = data.scales[-1]
            idx = np.argmin(np.abs(scale - number))
            return idx
        
        elif "hz" in value.lower():
            # Find nearest Hz
            scale = data.scales[-1]
            idx = np.argmin(np.abs(scale - number))
            return idx
        
        elif "pts" in value.lower():
            idx = number if number >= 0 else npoints + number
            return int(np.clip(idx, 0, npoints-1))
        
        elif "%" in value.lower():
            idx = (number / 100.0) * npoints
            idx = idx if idx >= 0 else npoints + idx
            return int(np.clip(idx, 0, npoints-1))
        
        else:
            # Assume pts if no unit
            idx = number if number >= 0 else npoints + number
            return int(np.clip(idx, 0, npoints-1))

    raise ValueError(f"Invalid start/end value: {value}")