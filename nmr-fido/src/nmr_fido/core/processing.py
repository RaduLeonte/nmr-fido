import numpy as np
import copy
from nmr_fido.nmrdata import NMRData


def fourier_transform(data: NMRData, axes: list[int] = None) -> NMRData:
    raise NotImplementedError
    return



def zero_fill(
    data: NMRData,
    *,
    factor: int = 1,
    add: int = None,
    final_size: int = None,
) -> NMRData:
    """
    Zero fill the last dimension of the data.

    Args:
        data (NMRData): Input data.
        factor (int, optional): How many times to double the size (2^factor). Default = 1 (double size once).
        add (int, optional): How many zeros to add to the last dimension.
        final_size (int, optional): Final size for the last dimension.

    Returns:
        NMRData: Zero-filled NMRData.
    """
    original_shape = list(data.shape)
    last_dim = original_shape[-1]
    
    
    last_unit = data.units[-1]
    if last_unit not in ("pts", None, "points"):
        raise ValueError(
            f"Cannot zero-fill: last dimension unit is '{last_unit}', expected 'pts' or None."
        )

    # If user sets anything other than factor, we switch mode
    if any(x is not None for x in (add, final_size)):
        if sum(x is not None for x in (add, final_size)) > 1:
            raise ValueError("Specify only one of 'add' or 'final_size'.")
        factor = None  # Ignore default doubling if add or final_size is given

    if factor is not None:
        new_last_dim = last_dim * (2 ** factor)
        method = 'factor'
    
    elif add is not None:
        new_last_dim = last_dim + add
        method = 'add'
    
    elif final_size is not None:
        if final_size < last_dim:
            raise ValueError(f"final_size {final_size} must be greater than current last dimension {last_dim}.")
        new_last_dim = final_size
        method = 'final_size'


    new_shape = original_shape[:-1] + [new_last_dim]

    # Create zero filled np.ndarray
    result_array = np.zeros(new_shape, dtype=data.dtype)

    # Copy original data into zero filled np.ndarray
    slicing = tuple(slice(0, s) for s in original_shape)
    result_array[slicing] = data

    result = NMRData(result_array, copy_from=data)
    
    # Update last scale with pts
    new_scales = list(result.scales)
    new_scales[-1] = np.arange(new_last_dim)
    result.scales = new_scales
    
    # Update processing history
    result.processing_history.append(
        {
            'Function': "Zero filling",
            'original_last_dim': last_dim,
            'new_last_dim': new_last_dim,
            'method': method
        }
    )
    return result
# NMRPipe alias
ZF = zero_fill