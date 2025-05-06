import numpy as np
import copy
from nmr_fido.nmrdata import NMRData
from nmr_fido.utils import _convert_to_index
from scipy.signal import hilbert


def solvent_filter(
    data: NMRData,
    #*,
    # Aliases
) -> NMRData:
    """
    Desc.

    Args:
        data (NMRData): Input data.

    Aliases:

    Returns:
        NMRData: .
    """
    
    raise NotImplementedError
    
    result = data
    
    return result

# NMRPipe alias
SOL = solvent_filter
SOL.__doc__ = solvent_filter.__doc__  # Auto-generated
SOL.__name__ = "SOL"  # Auto-generated



def linear_prediction(
    data: NMRData,
    *,
    # Aliases
    pred: int = None,
    x1: int = None,
    xn: int = None,
    ord: int = None,
    f: bool = None,
    b: bool = None,
    fb: bool = None,
    before: bool = None,
    after: bool = None,
    nofix: bool = None,
    fix: bool = None,
    fixMode: int = None,
    ps90_180: bool = None,
    ps0_0: bool = None,
    pca: bool = None,
    extra: int = None,
) -> NMRData:
    """
    Desc.

    Args:
        data (NMRData): Input data.

    Aliases:

    Returns:
        NMRData: .
    """
    
    raise NotImplementedError
    
    result = data
    
    return result

# NMRPipe alias
LP = linear_prediction
LP.__doc__ = linear_prediction.__doc__  # Auto-generated
LP.__name__ = "LP"  # Auto-generated



def sine_bell_window(
    data: NMRData,
    *,
    start_angle: float = 0.0,
    end_angle: float = 1.0,
    exponent: float = 1.0,
    size_window: int = None,
    start: int = 1,
    scale_factor_first_point: float = 1.0,
    fill_outside_one: bool = False,
    invert_window: bool = False,
    # Aliases
    off: float = None,
    end: float = None,
    pow: float = None,
    size: int = None,
    c: float = None,
    one: bool = None,
    inv: bool = None,
) -> NMRData:
    """
    Apply a sine-bell apodization (window) to the last dimension of the data.

    Args:
        data (NMRData): Input data.
        start_angle (float): Start of the sine bell in units of pi radians (default 0.0).
        end_angle (float): End of the sine bell in units of pi radians (default 1.0).
        exponent (float): Exponent applied to the sine bell (default 1.0).
        size_window (int, optional): Number of points in the window (default: size of last axis).
        start (int): Index to start applying the window (default 1 = first point).
        scale_factor_first_point (float): Scaling for the first point (default 1.0).
        fill_outside_one (bool): If True, data outside window is multiplied by 1.0 instead of 0.0.
        invert_window (bool): If True, apply 1/window instead of window and 1/scale_factor_first_point.

    Aliases:
        off: Alias for start_angle
        end: Alias for end_angle
        pow: Alias for exponent
        size: Alias for size_window
        c: Alias for scale_factor_first_point
        one: Alias for fill_outside_one
        inv: Alias for invert_window

    Returns:
        NMRData: Data after applying sine-bell apodization.
    """
    
    # Handle argument aliases
    if off is not None: start_angle = off
    if end is not None: end_angle = end
    if pow is not None: exponent = pow
    if size is not None: size_window = size
    if c is not None: scale_factor_first_point = c
    if one is not None: fill_outside_one = one
    if inv is not None: invert_window = inv
    
    if size_window is None:
        size_window = int(data.shape[-1])
    
    # Create window
    t = np.arange(size_window)
    window = np.power(
        np.sin(
            np.pi * start_angle + np.pi * (end_angle - start_angle) * t / (size_window - 1)
        ),
        exponent
    ).astype(data.dtype)
    
    # Invert window if necessary
    if invert_window:
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_window = np.zeros_like(window)
            mask = window != 0
            inv_window[mask] = 1.0 / window[mask]
            window = inv_window
            
        if scale_factor_first_point != 0.0:
            scale_factor_first_point = 1.0 / scale_factor_first_point
        else:
            scale_factor_first_point = 1.0
    
    # Create zeroes array to insert window function into
    full_window = np.zeros_like(data)
    npoints = data.shape[-1]
    
    if start - 1 >= npoints:
        raise ValueError(f"Start point {start} is beyond data size {npoints}.")
    
    # Clip window if it extends past the data length
    end_point = min((start - 1) + size_window, npoints)
    clip_size = end_point - (start - 1)
    
    # Insert window into array
    full_window[..., start-1:end_point] = window[:clip_size]
    
    # Multiply poitns outside the window range by 1
    if fill_outside_one:
        mask = full_window == 0
        full_window[mask] = 1.0
    
    # Apply window
    result = data * full_window
    
    # Scale first point
    result[..., start-1] *= scale_factor_first_point
    
    
    result.processing_history.append(
        {
            'Function': "Apodization: Sine bell window",
            'start_angle': start_angle,
            'end_angle': end_angle,
            'exponent': exponent,
            'size_window': size_window,
            'start': start,
            'scale_factor_first_point': scale_factor_first_point,
            'fill_outside_one': fill_outside_one,
            'invert_window': invert_window,
            'clip_size': clip_size,
        }
    )
    
    return result

# NMRPipe alias
SP = sine_bell_window
SP.__doc__ = sine_bell_window.__doc__  # Auto-generated
SP.__name__ = "SP"  # Auto-generated



def zero_fill(
    data: NMRData,
    *,
    factor: int = 1,
    add: int = None,
    final_size: int = None,
    # Aliases
    zf: int = None,
    pad: int = None,
    size: int = None,
) -> NMRData:
    """
    Zero fill the last dimension of the data.

    Args:
        data (NMRData): Input data.
        factor (int, optional): How many times to double the size (2^factor). Default = 1 (double size once).
        add (int, optional): How many zeros to add to the last dimension.
        final_size (int, optional): Final size for the last dimension.
        
    Aliases:
        zf: Alias for factor.
        pad: Alias for add.
        size: Alias for final_size.

    Returns:
        NMRData: Zero-filled NMRData.
    """
    # Handle argument aliases
    if zf is not None: factor = zf
    if pad is not None: add = pad
    if size is not None: final_size = size
    
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
ZF.__doc__ = zero_fill.__doc__  # Auto-generated
ZF.__name__ = "ZF"  # Auto-generated



def fourier_transform(
    data: NMRData,
    *,
    real_only: bool = False,
    inverse: bool = False,
    negate_imaginaries: bool = False,
    sign_alteration: bool = False,
    bruk: bool = False,
    #dmx: bool = False,
    #nodmx: bool = False,
    norm: str = "backward",
    # Aliases
    real: bool = None,
    inv: bool = None,
    neg: bool = None,
    alt: bool = None,
) -> NMRData:
    """
    Apply Fourier Transform to the last dimension of the NMRData.

    Args:
        data (NMRData): Input NMRData.
        real_only (bool): Promote real-only input to complex if True.
        inverse (bool): Perform inverse FFT if True.
        negate_imaginaries (bool): Multiply imaginary parts by -1 before FFT.
        sign_alteration (bool): Apply sign alternation to input (multiply every other point by -1).
        bruk (bool): If True, sets real_only and sign_alteration to True automatically (Bruker-style processing).
        norm (str): Normalization mode for the FFT
            - "backward" (default): No scaling on forward FFT, 1/N scaling on inverse FFT (standard NMR convention).
            - "forward": 1/N scaling on forward FFT, no scaling on inverse FFT (rare, signal processing style).
            - "ortho": √N scaling on both forward and inverse FFT (symmetric, orthonormal transform).

    Aliases:
        real: Alias for real_only.
        inv: Alias for inverse.
        neg: Alias for negate_imaginaries.
        alt: Alias for sign_alteration.

    Returns:
        NMRData: Fourier transformed data.
    """
    # Handle argument aliases
    if real is not None: real_only = real
    if inv is not None: inverse = inv
    if neg is not None: negate_imaginaries = neg
    if alt is not None: sign_alteration = alt
    
    if bruk:
        real_only = True
        sign_alteration = True

    array = data.copy()
    
    if np.isrealobj(array):
        if real_only:
            array = array.astype(np.complex128)
        else:
            raise ValueError(
                "Input data is real-only. Set real_only=True (or real=True) if you want to allow complex FT on real data."
            )

    # Sign alteration if needed
    if sign_alteration:
        alt = np.ones(array.shape[-1])
        alt[1::2] = -1
        array = array * alt

    # Negate imaginary parts if needed
    if negate_imaginaries:
        array = np.real(array) - 1j * np.imag(array)

    # Perform FFT or IFFT
    if inverse:
        transformed = np.fft.ifft(np.fft.ifftshift(array, axes=(-1,)), axis=-1, norm=norm)
    else:
        transformed = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(array, axes=(-1,)), axis=-1, norm=norm), axes=(-1,))


    # Result
    result = NMRData(transformed, copy_from=data)
    
    # Convert scale to ppm
    result.scale_to_ppm()

    # Update metadata
    result.processing_history.append(
        {
            'Function': 'Complex fourier transform',
            'real_only': real_only,
            'inverse': inverse,
            'negate_imaginaries': negate_imaginaries,
            'sign_alteration': sign_alteration,
            'bruk': bruk,
            'norm': norm,
            'input_real': np.isrealobj(data),
        }
    )

    return result

# NMRPipe alias
FT = fourier_transform
FT.__doc__ = fourier_transform.__doc__  # Auto-generated
FT.__name__ = "FT"  # Auto-generated



def hilbert_transform(
    data: NMRData,
    *,
    mirror_image: bool = False,
    temporary_zero_fill: bool = False,
    size_time_domain: int = None,
    # Aliases
    ps90_180: bool = None,
    zf: bool = None,
    td: bool = None,
) -> NMRData:
    """
    Apply a Hilbert transform to the last dimension of NMRData.

    Args:
        data (NMRData): Input data.
        mirror_image (bool): If True, use mirror image mode for HT (for P1=180 acquisitions).
        temporary_zero_fill (bool): If True, apply temporary zero filling for speed.
        size_time_domain (int, optional): Size of the time domain (half of original size for some data).

    Aliases:
        ps90_180: Alias for mirror_image.
        zf: Alias for temporary_zero_fill.
        td: Alias for size_time_domain.

    Returns:
        NMRData: Hilbert transformed data.
    """
    # Handle argument aliases
    if ps90_180 is not None: mirror_image = ps90_180
    if zf is not None: temporary_zero_fill = zf
    if td is not None: size_time_domain = td
    
    
    array = data.copy()
    
    original_shape = array.shape
    npoints = array.shape[-1]
    
    
    if size_time_domain is not None:
        npoints = size_time_domain
    
    
    if temporary_zero_fill:
        next_pow2 = 2**int(np.ceil(np.log2(npoints)))
        if next_pow2 > npoints:
            array = zero_fill(array, final_size=next_pow2)
    
    
    if mirror_image:
        mirrored = np.concatenate(
            (array, array[..., ::-1].conj()), axis=-1
        )
        hilbert_data = hilbert(mirrored, axis=-1)
        result_array = hilbert_data[..., :array.shape[-1]]
    else:
        # Standard Hilbert transform
        result_array = hilbert(array, axis=-1)


    # If temporary zero-filled, crop back
    if temporary_zero_fill:
        result_array = result_array[..., :original_shape[-1]]

    result = NMRData(result_array, copy_from=data)
    
    # Update metadata
    result.processing_history.append(
        {
            'Function': 'Hilbert transform',
            'mirror_image': mirror_image,
            'temporary_zero_fill': temporary_zero_fill,
            'size_time_domain': size_time_domain,
        }
    )

    return result

# NMRPipe alias
HT = hilbert_transform
HT.__doc__ = hilbert_transform.__doc__  # Auto-generated
HT.__name__ = "HT"  # Auto-generated



def phase(
    data: NMRData,
    *,
    p0: float = 0.0,
    p1: float = 0.0,
    invert: bool = False,
    reconstruct_imaginaries: bool = False,
    temporary_zero_fill: bool = False,
    exponential_correction: bool = False,
    decay_constant: float = 0.0,
    #right_shift_point_count: int = 0,
    #left_shift_point_count: int = 0,
    #sw: bool = False,
    # Aliases
    inv: float = None,
    ht: bool = None,
    zf: bool = None,
    exp: bool = None,
    tc: float = None,
    #rs: bool = None,
    #ls: bool = None,
    #sw: bool = None,
) -> NMRData:
    """
    Apply zero-order and first-order phase correction to the last dimension of the NMRData.

    Args:
        data (NMRData): Input data.
        p0 (float): Zero-order phase correction in degrees (constant shift).
        p1 (float): First-order phase correction in degrees across the sweep width.
        invert (bool): If True, apply the negative of the phase correction (e.g., for undoing previous phase).
        exponential_correction (bool): If True, apply exponential decay correction instead of linear first-order.
        decay_constant (float): Decay constant for exponential correction (only used if exponential_correction=True).
        reconstruct_imaginaries (bool): If True and data is real-only, reconstruct imaginary parts using Hilbert transform.
        temporary_zero_fill (bool): If True, temporarily zero-fill to next power of 2 for better phase smoothness.

    Aliases:
        inv: Alias for invert.
        ht: Alias for reconstruct_imaginaries.
        zf: Alias for temporary_zero_fill.
        exp: Alias for exponential_correction.
        tc: Alias for decay_constant.

    Returns:
        NMRData: Data after applying phase correction.
    """
    
    # Handle argument aliases
    if inv is not None: invert = inv
    if exp is not None: exponential_correction = exp
    if tc is not None: decay_constant = tc
    
    
    array = data.copy()

    # Hilbert transform if requested
    if reconstruct_imaginaries:
        array = hilbert_transform(np.real(array))
        
        
    original_shape = array.shape
    
    if temporary_zero_fill:
        npoints = array.shape[-1]
        next_pow2 = 2**int(np.ceil(np.log2(npoints)))
        if next_pow2 > npoints:
            array = zero_fill(array, final_size=next_pow2)
    
    npoints = array.shape[-1]
    x = np.arange(npoints)

    if exponential_correction:
        phase = np.deg2rad(p0 * np.exp(-decay_constant * x / npoints))
    else:
        phase = np.deg2rad(p0 + p1*( x / npoints))
    
    if invert:
        phase = -phase
    
    phase_correction = np.exp(1j * phase)
    
    result = array * phase_correction
    
    if temporary_zero_fill:
        result_array = result_array[..., :original_shape[-1]]
    
    result.processing_history.append(
        {
            'Function': "Phase Correction",
            'p0': p0,
            'p1': p1,
            'invert': invert,
            'exponential_correction': exponential_correction,
            'decay_constant': decay_constant,
            'reconstruct_imaginaries': reconstruct_imaginaries,
            'temporary_zero_fill': temporary_zero_fill,
        }
    )
    
    return result

# NMRPipe alias
PS = phase
PS.__doc__ = phase.__doc__  # Auto-generated
PS.__name__ = "PS"  # Auto-generated



def extract_region(
    data: NMRData,
    *,
    start: str | int = None,
    end: str | int = None,
    start_y: str | int = None,
    end_y: str | int = None,
    left_half: bool = False,
    right_half: bool = False,
    middle_half: bool = False,
    power_of_two: bool = False,
    multiple_of: int = None,
    # Aliases
    #time: bool = None,
    left: bool = None,
    right: bool = None,
    mid: bool = None,
    pow2: bool = None,
    #sw: bool = None,
    round: int = None,
    x1: str = None,
    xn: str = None,
    y1: str = None,
    yn: str = None,
) -> NMRData:
    """
    Desc.

    Args:
        data (NMRData): The data to extract region from.

    Returns:
        NMRData: Desc.
    """
    # Handle aliases
    if left is not None: left_half = left
    if right is not None: right_half = right
    if mid is not None: middle_half = mid
    if pow2 is not None: power_of_two = pow2
    if round is not None: multiple_of = round
    if x1 is not None: start = x1
    if xn is not None: end = xn
    if y1 is not None: start_y = y1
    if yn is not None: end_y = yn
    
    array = data.copy()
    npoints = array.shape[-1]
    
    # Determine start and end indices
    start_idx = _convert_to_index(data, start, npoints, default=0)
    end_idx = _convert_to_index(data, end, npoints, default=npoints-1)

    # Ensure valid index order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    raise NotImplementedError
    
    result = data

    result.processing_history.append(
        {
            'Function': "Extract region",
        }
    )

    return result

# NMRPipe alias
EXT = extract_region
EXT.__doc__ = extract_region.__doc__  # Auto-generated
EXT.__name__ = "EXT"  # Auto-generated



def polynomial_baseline_correction(
    data: NMRData,
    *,
    domain: str = "freq",
    order: int = 4,
    # Aliases
    sx1: int = None,
    sxn: int = None,
    fx1: int = None,
    fxn: int = None,
    x1: int = None,
    xn: int = None,
    nl: int = None,
    nw: int = None,
    ord: int = None,
    nc: int = None,
    first: int = None,
    last: int = None,
    avg: int = None,
    filt: int = None,
    time: int = None,
) -> NMRData:
    """
    Desc.

    Args:
        data (NMRData): The data to transpose.

    Returns:
        NMRData: .
    """
    
    raise NotImplementedError
    
    result = data
    
    result.processing_history.append(
        {
            'Function': "Polynomial baseline correction",
        }
    )
    
    return result

# NMRPipe alias
POLY = polynomial_baseline_correction
POLY.__doc__ = polynomial_baseline_correction.__doc__  # Auto-generated
POLY.__name__ = "POLY"  # Auto-generated



def transpose(
    data: NMRData,
    *,
    axes: list[int] = None
    #hyper: bool = False,
) -> NMRData:
    """
    Transpose the data and reorder metadata accordingly.

    Args:
        data (NMRData): The data to transpose.
        axes (list[int], optional): New axis order. If None, reverse axes.

    Returns:
        NMRData: Transposed data.
    """

    if axes:
        axes = axes[0]
    else:
        axes = reversed(range(data.ndim))
    
    result = super(NMRData, data).transpose(*axes)

    for attr in data._custom_attrs:
        match attr:
            case 'scales' | 'labels' | 'units' | 'axis_info':
                setattr(result, attr, [getattr(data, attr)[ax] for ax in axes])
            case _:
                setattr(result, attr, copy.deepcopy(getattr(data, attr)))

    # Record processing history
    result.processing_history.append(
        {
            'Function': "Transpose",
            'axes': list(axes) if hasattr(axes, '__iter__') else [axes],
            'shape_before': data.shape,
            'shape_after': result.shape,
        }
    )

    return result

# NMRPipe alias
TP = transpose
TP.__doc__ = transpose.__doc__  # Auto-generated
TP.__name__ = "TP"  # Auto-generated

ZTP = transpose
ZTP.__doc__ = transpose.__doc__  # Auto-generated
ZTP.__name__ = "ZTP"  # Auto-generated


def add_constant(
    data: NMRData,
    *,
    start: str | int = None,
    end: str | int = None,
    constant: float = None,
    constant_real: float = None,
    constant_imaginary: float = None,
    # Alias
    r: float = None,
    i: float = None,
    c: float = None,
    x1: str | int = None,
    xn: str | int = None,
) -> NMRData:
    """
    Add a constant with the NMRData.

    Args:
        data (NMRData): Input data.
        start (str or int, optional): Start point or coordinate ("5.5 ppm", "1234 pts", etc.).
        end (str or int, optional): End point or coordinate.
        constant (float): Real value to add to both real and imaginary parts of data.
        constant_real (float): Real value to add to real part only.
        constant_imaginary (float): Real value to add to imaginary part only.

    Aliases:
        r: Alias for constant_real.
        i: Alias for constant_imaginary.
        c: Alias for constant.
        x1: Alias for start.
        xn: Alias for end.

    Returns:
        NMRData: Adjusted data.
    """
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn
    
    
    if constant is None and constant_real is None and constant_imaginary is None:
        raise ValueError("At least one of 'constant', 'constant_real', or 'constant_imaginary' must be specified.")
    
    
    array = data.copy()

    npoints = array.shape[-1]
    
    
    # Determine start and end indices
    start_idx = _convert_to_index(data, start, npoints, default=0)
    end_idx = _convert_to_index(data, end, npoints, default=npoints-1)
    
    
    # Ensure valid index order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx


    slice_obj = (slice(None),) * (array.ndim - 1) + (slice(start_idx, end_idx + 1),)

    if np.iscomplexobj(array):
        # Complex data
        if constant is not None:
            array[slice_obj] += constant
        
        if constant_real is not None:
            array.real[slice_obj] += constant_real
        
        if constant_imaginary is not None:
            array.imag[slice_obj] += constant_imaginary
    else:
        # Real-only data
        if constant_imaginary is not None:
            raise ValueError("Cannot multiply imaginary part on real-only data.")
        
        if constant_real is not None:
            array[..., start_idx:end_idx+1] += constant_real
        
        elif constant is not None:
            array[..., start_idx:end_idx+1] += constant

    result = NMRData(array, copy_from=data)


    result.processing_history.append(
        {
            'Function': "Add constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
        }
    )

    return result

# NMRPipe alias
ADD = add_constant
ADD.__doc__ = add_constant.__doc__  # Auto-generated
ADD.__name__ = "ADD"  # Auto-generated


def multiply_constant(
    data: NMRData,
    *,
    start: str | int = None,
    end: str | int = None,
    constant: float = None,
    constant_real: float = None,
    constant_imaginary: float = None,
    # Alias
    r: float = None,
    i: float = None,
    c: float = None,
    x1: str | int = None,
    xn: str | int = None,
) -> NMRData:
    """
    Multiply a constant with the NMRData.

    Args:
        data (NMRData): Input data.
        start (str or int, optional): Start point or coordinate ("5.5 ppm", "1234 pts", etc.).
        end (str or int, optional): End point or coordinate.
        constant (float): Real value to multiply both real and imaginary parts of data.
        constant_real (float): Real value to multiply real part only.
        constant_imaginary (float): Real value to multiply imaginary part only.

    Aliases:
        r: Alias for constant_real.
        i: Alias for constant_imaginary.
        c: Alias for constant.
        x1: Alias for start.
        xn: Alias for end.

    Returns:
        NMRData: Adjusted data.
    """
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn
    
    
    if constant is None and constant_real is None and constant_imaginary is None:
        raise ValueError("At least one of 'constant', 'constant_real', or 'constant_imaginary' must be specified.")
    
    
    array = data.copy()

    npoints = array.shape[-1]
    
    
    # Determine start and end indices
    start_idx = _convert_to_index(data, start, npoints, default=0)
    end_idx = _convert_to_index(data, end, npoints, default=npoints-1)
    
    
    # Ensure valid index order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx


    slice_obj = (slice(None),) * (array.ndim - 1) + (slice(start_idx, end_idx + 1),)

    if np.iscomplexobj(array):
        # Complex data
        if constant is not None:
            array[slice_obj] *= constant
        
        if constant_real is not None:
            array.real[slice_obj] *= constant_real
        
        if constant_imaginary is not None:
            array.imag[slice_obj] *= constant_imaginary
    else:
        # Real-only data
        if constant_imaginary is not None:
            raise ValueError("Cannot multiply imaginary part on real-only data.")
        
        if constant_real is not None:
            array[..., start_idx:end_idx+1] *= constant_real
        
        elif constant is not None:
            array[..., start_idx:end_idx+1] *= constant

    result = NMRData(array, copy_from=data)


    result.processing_history.append(
        {
            'Function': "Multiply constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
        }
    )

    return result

# NMRPipe alias
MULT = multiply_constant
MULT.__doc__ = multiply_constant.__doc__  # Auto-generated
MULT.__name__ = "MULT"  # Auto-generated


def set_to_constant(
    data: NMRData,
    *,
    start: str | int = None,
    end: str | int = None,
    constant: float = 0.0,
    constant_real: float = 0.0,
    constant_imaginary: float = 0.0,
    # Alias
    r: float = None,
    i: float = None,
    c: float = None,
    x1: str | int = None,
    xn: str | int = None,
) -> NMRData:
    """
    Set a range in the last dimension of NMRData to a constant.

    Args:
        data (NMRData): Input data.
        start (str or int, optional): Start point or coordinate ("5.5 ppm", "1234 pts", etc.).
        end (str or int, optional): End point or coordinate.
        constant (float): Real value to set (applies to real part if complex).
        constant_real (float): Real part to set (for complex data).
        constant_imaginary (float): Imaginary part to set (for complex data).

    Aliases:
        r: Alias for constant_real.
        i: Alias for constant_imaginary.
        c: Alias for constant.
        x1: Alias for start.
        xn: Alias for end.

    Returns:
        NMRData: Adjusted data.
    """
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn

    if constant is None and constant_real is None and constant_imaginary is None:
        raise ValueError("At least one of 'constant', 'constant_real', or 'constant_imaginary' must be specified.")

    array = data.copy()
    npoints = array.shape[-1]

    # Determine start and end indices
    start_idx = _convert_to_index(data, start, npoints, default=0)
    end_idx = _convert_to_index(data, end, npoints, default=npoints-1)

    # Ensure valid index order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    slice_obj = (slice(None),) * (array.ndim - 1) + (slice(start_idx, end_idx + 1),)

    if np.iscomplexobj(array):
        if constant is not None:
            array[slice_obj] = constant
        
        if constant_real is not None:
            array.real[slice_obj] = constant_real
        
        if constant_imaginary is not None:
            array.imag[slice_obj] = constant_imaginary
    else:
        # Real-only data
        if constant_imaginary is not None:
            raise ValueError("Cannot set imaginary part on real-only data.")
        
        if constant_real is not None:
            array[..., start_idx:end_idx+1] = constant_real
        
        elif constant is not None:
            array[..., start_idx:end_idx+1] = constant

    result = NMRData(array, copy_from=data)

    result.processing_history.append(
        {
            'Function': "Set to constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
        }
    )

    return result

# NMRPipe alias
SET = set_to_constant
SET.__doc__ = set_to_constant.__doc__  # Auto-generated
SET.__name__ = "SET"  # Auto-generated