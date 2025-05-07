import time
import numpy as np
import copy
from nmr_fido.nmrdata import NMRData
from nmr_fido.utils import _convert_to_index, get_ppm_scale
from scipy.signal import hilbert
from scipy import signal


def _format_elapsed_time(elapsed: float) -> str:
    """Format elapsed time to a human readable string for the NMRData processing history.

    Args:
        elapsed (float): Elapsed time in s

    Returns:
        str: Formatted elapsed time
    """
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed % 1) * 1000)
    microseconds = int((elapsed % 1) * 1_000_000) % 1000

    if minutes > 0:
        return f"{minutes}m {seconds}s {milliseconds}ms {microseconds}µs"

    return f"{seconds}s {milliseconds}ms {microseconds}µs"


def _interleaved_to_complex(data: NMRData, dim: int = -1) -> 'NMRData':
    """Convert interleaved data [re1, im1, re2, im2, ...] to complex data [re1 + 1j*im1, re2 + 1j*im2, ....]

    Args:
        data (NMRData): Input data.
        dim (int, optional): Target dimension to convert. Defaults to -1.


    Returns:
        NMRData: Data after convertion.
    """
    # Normalize the axis to handle negative indices
    dim = dim if dim >= 0 else data.ndim + dim
    
    # Calculate the new shape, halving the last dimension
    new_shape = list(data.shape)
    if new_shape[dim] % 2 != 0:
        raise ValueError(
            f"The target axis {dim} length ({data.shape[dim]}) must be even, representing interleaved real/imaginary pairs."
        )

    new_shape[dim] //= 2

    # Rearrange the data along the target axis
    slices_real = [slice(None)] * data.ndim
    slices_imag = [slice(None)] * data.ndim

    slices_real[dim] = slice(0, None, 2)  # Real parts
    slices_imag[dim] = slice(1, None, 2)  # Imaginary parts

    # Construct the complex array
    complex_dtype = np.result_type(data, np.complex64)
    complex_data = np.empty(new_shape, dtype=complex_dtype)
    complex_data.real = data[tuple(slices_real)]
    complex_data.imag = data[tuple(slices_imag)]
    
    result = NMRData(complex_data, copy_from=data)
    result.axes[dim]["acqu_mode"] = "Complex"

    return result


def solvent_filter(
    data: NMRData,
    *,
    filter_mode: str = "Low Pass",
    lowpass_size: int = 16,
    lowpass_shape: str = "Boxcar",
    butter_ord: int = 4,
    butter_cutoff: float = 0.05,
    poly_ext_order: int = 2,
    spline_noise: float = 1.0,
    smooth_factor: float = 1.1,
    skip_points: int = 0,
    use_poly_ext: bool = True,
    use_mirror_ext: bool = False,
    
    # Aliases
    mode: str = None,
    fl: int = None,
    fs: int = None,
    po: int = None,
    sn: float = None,
    sf: float = None,
    head: int = None,
    poly: bool = None,
    mir: bool = None,
    #noseq: bool = None,
    #nodms: bool = None,
) -> NMRData:
    """
    Desc.

    Args:
        data (NMRData): Input data.

    Aliases:

    Returns:
        NMRData: .
    """
    start_time = time.perf_counter()
    
    # Handle argument aliases
    if mode is not None: filter_mode = {1: "Low Pass", 2: "Spline", 3: "Polynomial"}[mode]
    if fl is not None: lowpass_size = fl
    if fs is not None: lowpass_shape = {1: "Boxcar", 2: "Sine", 3: "Sine^2"}[fs]
    if po is not None: poly_ext_order = po
    if sn is not None: spline_noise = sn
    if sf is not None: smooth_factor = sf
    if head is not None: skip_points = head
    if poly is not None: use_poly_ext = poly
    if mir is not None: use_mirror_ext = mir


    result = data.copy()
    sliced_data = result[..., skip_points:]
    
    filter_width = lowpass_size*2 + 1

    match filter_mode:
        case "Low Pass":
            filter = None
            match lowpass_shape:
                case "Boxcar":
                    filter = np.ones(filter_width, float)
                
                case "Sine":
                    filter = np.cos(np.pi * np.linspace(-0.5, 0.5, filter_width))
                
                case "Sine^2":
                    filter = np.cos(np.pi * np.linspace(-0.5, 0.5, filter_width)) ** 2
                    
                case "Butterworth":
                    b, a = signal.butter(butter_ord, butter_cutoff, btype='low', analog=False)
            
            if filter is not None:
                for index in np.ndindex(sliced_data.shape[:-1]):
                    fid = sliced_data[index]
                    # Apply convolution
                    filtered_fid = signal.convolve(fid, filter, mode="same") / filter_width
                    # Subtract the filtered signal from the original
                    sliced_data[index] = fid - filtered_fid
                    
            elif lowpass_shape == "Butterworth":
                for index in np.ndindex(sliced_data.shape[:-1]):
                    fid = sliced_data[index]
                    # Apply Butterworth filter
                    filtered_fid = signal.filtfilt(b, a, fid)
                    # Subtract the filtered signal from the original
                    sliced_data[index] = fid - filtered_fid

            pass
        
        case "Spline":
            raise NotImplementedError
        
        case "Polynomial":
            raise NotImplementedError


    result[..., skip_points:] = sliced_data
    
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Solvent filter",
            'filter_mode': filter_mode,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        }
    )
    
    return result

# NMRPipe alias
SOL = solvent_filter
SOL.__doc__ = solvent_filter.__doc__  # Auto-generated
SOL.__name__ = "SOL"  # Auto-generated



def linear_prediction(
    data: NMRData,
    *,
    prediction_size: int = -1,
    fit_start: int = 0,
    fit_end: int = -1,
    order: int = 8,
    direction: str = "forward",
    use_root_fixing: bool = False,
    root_fix_mode: int = "suppress_increasing",
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
    # Handle argument aliases
    if pred is not None: prediction_size = pred
    if x1 is not None: fit_start = x1
    if xn is not None: fit_end = xn
    if ord is not None: order = ord
    if f: direction = "forward"
    if b: direction = "backward"
    if fb: direction = "both"
    if before is not None: direction = "backward"
    if after is not None: direction = "forward"
    if nofix: use_root_fixing = False
    if fix: use_root_fixing = True
    if fixMode is not None: root_fix_mode = {-1: "suppress_decreasing", 0: None, 1: "suppress_increasing"}[fixMode]
    
    
    result = data.copy()
    npoints = result.shape[-1]
    
    if order >= npoints/2:
        raise ValueError(f"Number of coefficients ({order=}) must be less than half the number of points in the vector ({npoints=})")
    
    if prediction_size == -1: prediction_size = npoints
    
    fit_start_idx = _convert_to_index(result, fit_start, npoints, default=0)
    fit_end_idx = _convert_to_index(result, fit_end, npoints, default=npoints - 1)
    
    
    def fit_coeff(vector: np.ndarray) -> np.ndarray:
        x_fit = np.arange(fit_start_idx, fit_end_idx + 1)
        y_fit = vector[fit_start_idx:fit_end_idx + 1]
        
        coeff = None
        
        return coeff
    
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
    start_time = time.perf_counter()
    
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
    
    elapsed = time.perf_counter() - start_time
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
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        }
    )
    
    return result

# NMRPipe alias
SP = sine_bell_window
SP.__doc__ = sine_bell_window.__doc__  # Auto-generated
SP.__name__ = "SP"  # Auto-generated


def lorentz_to_gauss_window(
    data: NMRData,
    *,
    inv_exp_width: float = 0.0,
    broaden_width: float = 0.0,
    center: float = 0.0,
    size_window: int = None,
    start: int = 1,
    scale_factor_first_point: float = 1.0,
    fill_outside_one: bool = False,
    invert_window: bool = False,
    # Aliases
    g1: float = None,
    g2: float = None,
    g3: float = None,
    size: int = None,
    c: float = None,
    one: bool = None,
    inv: bool = None,
) -> NMRData:
    """
    Apply a Lorentz-to-Gauss apodization (window) to the last dimension of the data.

    Args:
        data (NMRData): Input data.
        inv_exp_width (float): Inverse exponential width (default 0.0).
        broaden_width (float): Broadening width for Gaussian function (default 0.0).
        center (float): Center point of the Gaussian function (default 0.0).
        size_window (int, optional): Number of points in the window (default: size of last axis).
        start (int): Index to start applying the window (default 1 = first point).
        scale_factor_first_point (float): Scaling for the first point (default 1.0).
        fill_outside_one (bool): If True, data outside window is multiplied by 1.0 instead of 0.0.
        invert_window (bool): If True, apply 1/window instead of window and 1/scale_factor_first_point.

    Aliases:
        g1: Alias for inv_exp_width
        g2: Alias for broaden_width
        g3: Alias for center
        size: Alias for size_window
        c: Alias for scale_factor_first_point
        one: Alias for fill_outside_one
        inv: Alias for invert_window

    Returns:
        NMRData: Data after Lorentz-to-Gauss apodization.
    """
    start_time = time.perf_counter()
    
    # Handle argument aliases
    if g1 is not None: inv_exp_width = g1
    if g2 is not None: broaden_width = g2
    if g3 is not None: center = g3
    if size is not None: size_window = size
    if c is not None: scale_factor_first_point = c
    if one is not None: fill_outside_one = one
    if inv is not None: invert_window = inv
    
    if size_window is None:
        size_window = int(data.shape[-1])
    
    sw = data.axes[-1].get("SW", None)
    if sw is None:
        raise ValueError("Spectral width (SW) is not defined in the data axis.")
    
    # Create window
    t = np.arange(size_window)
    npoints = data.shape[-1]
    center_index = int(center * (npoints - 1))
    
    exp_component = np.exp((np.pi * t * inv_exp_width) / sw)
    gauss_component = np.exp(
        -((0.6 * np.pi * broaden_width * (center_index - t)) ** 2)
    )
    window = (exp_component * gauss_component).astype(data.dtype)
    
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
    if scale_factor_first_point != 1.0:
        result[..., start - 1] *= scale_factor_first_point
    
    
    elapsed = time.perf_counter() - start_time
    result.processing_history.append({
        'Function': "Apodization: Lorentz-to-Gauss window",
        'inv_exp_width': inv_exp_width,
        'broaden_width': broaden_width,
        'center': center,
        'size_window': size_window,
        'start': start,
        'scale_factor_first_point': scale_factor_first_point,
        'fill_outside_one': fill_outside_one,
        'invert_window': invert_window,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })
    
    return result

# NMRPipe alias
GM = lorentz_to_gauss_window
GM.__doc__ = lorentz_to_gauss_window.__doc__  # Auto-generated
GM.__name__ = "GM"  # Auto-generated



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
    start_time = time.perf_counter()
    
    # Handle argument aliases
    if zf is not None: factor = zf
    if pad is not None: add = pad
    if size is not None: final_size = size
    
    original_shape = list(data.shape)
    last_dim = original_shape[-1]
    
    
    last_unit = data.axes[-1]["unit"]
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
    result.axes[-1]["scale"] = np.arange(new_last_dim)
    
    # Update processing history
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Zero filling",
            'original_last_dim': last_dim,
            'new_last_dim': new_last_dim,
            'method': method,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
        real_only (bool): Set imaginary part of data to 0 before performing FFT.
        inverse (bool): Perform inverse FFT if True.
        negate_imaginaries (bool): Multiply imaginary parts by -1 before FFT.
        sign_alteration (bool): Apply sign alternation to input (multiply every other point by -1).
        bruk (bool): If True, sets real_only and sign_alteration to True automatically (Bruker-style processing).

    Aliases:
        real: Alias for real_only.
        inv: Alias for inverse.
        neg: Alias for negate_imaginaries.
        alt: Alias for sign_alteration.

    Returns:
        NMRData: Fourier transformed data.
    """
    start_time = time.perf_counter()
    
    # Handle argument aliases
    if real is not None: real_only = real
    if inv is not None: inverse = inv
    if neg is not None: negate_imaginaries = neg
    if alt is not None: sign_alteration = alt
    
    if bruk:
        real_only = True
        sign_alteration = True

    array = data.copy()
    
    if real_only:
        array = array.astype(np.complex128)
        array.imag = 0.0

    # Sign alteration if needed
    if sign_alteration and not inverse:
        # Sign alteration for inverse is applied after ifft
        alternating_ones = np.ones(array.shape[-1])
        alternating_ones[1::2] = -1
        array = array * alternating_ones

    # Negate imaginary parts if needed
    if negate_imaginaries:
        array = np.real(array) - 1j * np.imag(array)

    if not np.iscomplex(array).all():
        array = array.astype("complex64")

    # Perform FFT or IFFT
    if inverse:
        transformed = (
            np.fft.fft(
                np.fft.ifftshift(array, axes=(-1,)),
                axis=-1
            ).astype(data.dtype)
        )
        # Data comes out as data * 1 because we're using fft for inverse FT
        # but we need data * 1/N
        transformed /= int(data.shape[-1]) # apply norm
        
        if sign_alteration:
            alternating_ones = np.ones(array.shape[-1])
            alternating_ones[1::2] = -1
            transformed = transformed * alternating_ones
        
    else:
        transformed = (
            np.fft.fftshift(
                np.fft.ifft(array, axis=-1).astype(data.dtype),
                axes=(-1,),
            )
        )
        # Data comes out as data * 1/N because we're using ifft for normal FT
        transformed *= int(data.shape[-1]) # undo norm


    # Result
    result = NMRData(transformed, copy_from=data)
    
    
    # Convert scale to ppm
    result.scale_to_ppm()

    # Update metadata
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': 'Complex fourier transform',
            'real_only': real_only,
            'inverse': inverse,
            'negate_imaginaries': negate_imaginaries,
            'sign_alteration': sign_alteration,
            'bruk': bruk,
            'input_real': np.isrealobj(data),
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    start_time = time.perf_counter()
    
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
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': 'Hilbert transform',
            'mirror_image': mirror_image,
            'temporary_zero_fill': temporary_zero_fill,
            'size_time_domain': size_time_domain,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    # TO DO: Implement time domain phase correction
    start_time = time.perf_counter()
    
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
    
    elapsed = time.perf_counter() - start_time
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
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    adjust_spectral_width: bool = True,
    multiple_of: int = None,
    # Aliases
    #time: bool = None,
    left: bool = None,
    right: bool = None,
    mid: bool = None,
    pow2: bool = None,
    sw: bool = None,
    round: int = None,
    x1: str = None,
    xn: str = None,
    y1: str = None,
    yn: str = None,
) -> NMRData:
    """
    Extract a region from the NMRData array, optionally adjusting spectral calibration.

    Args:
        data (NMRData): Input NMR dataset to extract from.
        start (str | int, optional): Starting point along the last dimension. Accepts index or unit string (e.g. "5.5 ppm", "1000 pts").
        end (str | int, optional): Ending point along the last dimension. Accepts index or unit string.
        start_y (str | int, optional): Starting vector along second-to-last dimension (for 2D data).
        end_y (str | int, optional): Ending vector along second-to-last dimension (for 2D data).
        left_half (bool): Extract the left half of the x-dimension.
        right_half (bool): Extract the right half of the x-dimension.
        middle_half (bool): Extract the middle half of the x-dimension.
        power_of_two (bool): Adjust extracted size to the nearest power of two.
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata based on extracted region.
        multiple_of (int, optional): Round the size to the nearest multiple of this value.
        
    Aliases:
        left, right, mid: Aliases for left_half, right_half, middle_half.
        pow2: Alias for power_of_two.
        sw: Alias for adjust_spectral_width.
        round: Alias for multiple_of.
        x1, xn: Aliases for start and end.
        y1, yn: Aliases for start_y and end_y.


    Returns:
        NMRData: Extracted region of the input data, optionally with updated spectral calibration.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if left is not None: left_half = left
    if right is not None: right_half = right
    if mid is not None: middle_half = mid
    if pow2 is not None: power_of_two = pow2
    if sw is not None: adjust_spectral_width = sw
    if round is not None: multiple_of = round
    if x1 is not None: start = x1
    if xn is not None: end = xn
    if y1 is not None: start_y = y1
    if yn is not None: end_y = yn
    
    result = data.copy()
    npoints = result.shape[-1]
    nvectors = result.shape[-2] if result.ndim > 1 else 1
    
    if left_half:
        start, end = 0, npoints // 2 - 1
    elif right_half:
        start, end = npoints // 2, npoints - 1
    elif middle_half:
        start = npoints // 4
        end = start + npoints // 2 - 1
        
    if start is None: start = 0
    if end is None: end = npoints - 1
    if start_y is None and result.ndim > 1: start_y = 0
    if end_y is None and result.ndim > 1: end_y = nvectors - 1
    
    # Convert coordinate-like strings to indices
    start_idx = _convert_to_index(result, start, npoints, default=0)
    end_idx = _convert_to_index(result, end, npoints, default=npoints - 1)
    if result.ndim > 1:
        start_y_idx = _convert_to_index(result, start_y, nvectors, default=0, dim=-2)
        end_y_idx = _convert_to_index(result, end_y, nvectors, default=nvectors - 1, dim=-2)
    else:
        start_y_idx = end_y_idx = 0

    # Ensure proper ordering
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    if start_y_idx > end_y_idx:
        start_y_idx, end_y_idx = end_y_idx, start_y_idx

    # Calculate slice sizes
    x_size = end_idx - start_idx + 1
    y_size = end_y_idx - start_y_idx + 1
    
    # Adjust size if rounding requested
    if power_of_two:
        new_x_size = 2 ** int(np.floor(np.log2(x_size)))
        end_idx = start_idx + new_x_size - 1
    elif multiple_of:
        new_x_size = x_size - (x_size % multiple_of)
        end_idx = start_idx + new_x_size - 1
        
    # Slice the data
    if result.ndim > 1:
        sliced = result[start_y_idx:end_y_idx+1, start_idx:end_idx+1]
    else:
        sliced = result[start_idx:end_idx+1]
    
    new_data = NMRData(sliced, copy_from=result)
    
    if adjust_spectral_width:
        dim = -1 
        full_size = npoints
        new_size = sliced.shape[dim]

        # Adjust SW, ORI, and OBS based on ppm limits
        sw, ori, obs = (data.axes[dim][k] for k in ("SW", "ORI", "OBS"))
        
        # Recalculate ppm scale and determine new max ppm based on trimmed size
        ppm_scale = get_ppm_scale(full_size, sw, ori, obs)
        new_ppm_scale = ppm_scale[start_idx:end_idx+1]
        ppm_min, ppm_max = new_ppm_scale.min(), new_ppm_scale.max()

        # Create new axis info dictionary
        new_axis_dict = result.axes[dim].copy()
        new_axis_dict['SW'] = sw * (new_size / full_size)
        new_axis_dict['ORI'] = obs * ppm_max  # ORI = center frequency in Hz = OBS * ppm
        new_axis_dict['OBS'] = obs  # unchanged
        new_axis_dict['scale'] = new_ppm_scale

        new_data.axes[dim] = new_axis_dict


    elapsed = time.perf_counter() - start_time
    new_data.processing_history.append({
        'Function': "Extract Region",
        'start_x': start_idx,
        'end_x': end_idx,
        'start_y': start_y_idx,
        'end_y': end_y_idx,
        'shape_before': result.shape,
        'shape_after': sliced.shape,
        'adjusted_sw': adjust_spectral_width,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return new_data

# NMRPipe alias
EXT = extract_region
EXT.__doc__ = extract_region.__doc__  # Auto-generated
EXT.__name__ = "EXT"  # Auto-generated



def polynomial_baseline_correction(
    data: NMRData,
    *,
    sub_start: int = 0,
    sub_end: int = -1,
    fit_start: int = 0,
    fit_end: int = -1,
    start: int = None,
    end: int = None,
    node_list: list[int] = None,
    node_width: int = 1,
    order: int = 4,
    initial_fit_nodes: int = 0,
    use_first_points: bool = False,
    use_last_points: bool = False,
    use_node_avg: bool = False,
    sine_filter: bool = False,
    
    domain: str = "frequency",
    noise_window_size: int = 8,
    min_baseline_fraction: float = 0.33,
    noise_adjustment_factor: float = 1.5,
    rms_noise_value: float = 0.0,
    
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
    
    time: bool = None,
    window: int = None,
    frac: float = None,
    nf: float = None,
    noise: float = None,
    #noseq: bool = None,
    #nodmx: bool = None,
) -> NMRData:
    """
    Apply polynomial baseline correction to the last dimension of the NMRData.

    Args:
        data (NMRData): Input data.
        sub_start (int): Start index for baseline subtraction region. Defaults to 0.
        sub_end (int): End index for baseline subtraction region. Defaults to the last index.
        fit_start (int): Start index for baseline fitting region. Defaults to 0.
        fit_end (int): End index for baseline fitting region. Defaults to the last index.
        start (int, optional): If provided, overrides both `sub_start` and `fit_start`.
        end (int, optional): If provided, overrides both `sub_end` and `fit_end`.
        node_list (list[int], optional): List of node center indices for fitting. If not specified, automatic node selection may be applied.
        node_width (int): Number of points to include on each side of each node center for fitting. Defaults to 1.
        order (int): Polynomial order for baseline fitting. Defaults to 4.
        initial_fit_nodes (int): Number of initial nodes to include in the fit. If 0, no initial nodes are used.
        use_first_points (bool): If True, include the first few points as nodes for baseline fitting. Defaults to False.
        use_last_points (bool): If True, include the last few points as nodes for baseline fitting. Defaults to False.
        use_node_avg (bool): If True, use average values within nodes instead of individual points for fitting. Defaults to False.
        sine_filter (bool): If True, apply a sine filter to the node data. Requires `use_node_avg` to be True. Defaults to False.
        
        domain (str): Data domain, either "frequency" or "time". Defaults to "frequency".
        noise_window_size (int): Window size for noise estimation. Only applicable in the "time" domain.
        min_baseline_fraction (float): Minimum fraction of data to consider as baseline. Only applicable in the "time" domain.
        noise_adjustment_factor (float): Adjustment factor for noise thresholding. Only applicable in the "time" domain.
        rms_noise_value (float): Pre-computed RMS noise value. Only applicable in the "time" domain.

    Aliases:
        sx1: Alias for `sub_start`.
        sxn: Alias for `sub_end`.
        fx1: Alias for `fit_start`.
        fxn: Alias for `fit_end`.
        x1: Alias for `start`.
        xn: Alias for `end`.
        nl: Alias for `node_list`.
        nw: Alias for `node_width`.
        ord: Alias for `order`.
        nc: Alias for `initial_fit_nodes`.
        first: Alias for `use_first_points`.
        last: Alias for `use_last_points`.
        avg: Alias for `use_node_avg`.
        filt: Alias for `sine_filter`.
        time: If True, switch domain to "time" and enable time-domain processing parameters.
        window: Alias for `noise_window_size`.
        frac: Alias for `min_baseline_fraction`.
        nf: Alias for `noise_adjustment_factor`.
        noise: Alias for `rms_noise_value`.
    
    
    Unimplemented/Unused Arguments:
        - initial_fit_nodes
        - use_first_points
        - use_last_points
        - use_node_avg
        - sine_filter

    Returns:
        NMRData: Data after applying polynomial baseline correction.
    """
    # TO DO: Implement missing arguments
    start_time = time.perf_counter()
    
    # Switch domain
    if time is not None: domain = "time"
    
    if domain == "time":
        # Handle aliases
        if window is not None: noise_window_size = window
        if frac is not None: min_baseline_fraction = frac
        if nf is not None: noise_adjustment_factor = nf
        if noise is not None: rms_noise_value = noise
        
        npoints = data.shape[-1]
        window_size = noise_window_size
        min_baseline_pts = int(min_baseline_fraction * npoints)
        
        # Estimate RMS noise value
        if rms_noise_value == 0.0:
            # Divide data into non-overlapping windows for each slice along the last dimension
            windows = [data[..., i:i + window_size] for i in range(0, npoints, window_size)]
            noise_estimates = [np.std(w, axis=-1) for w in windows if w.shape[-1] == window_size]
            
            if noise_estimates:
                # Compute median RMS noise value across all windows
                rms_noise_value = np.median(np.concatenate(noise_estimates), axis=-1)
        
        # Define baseline threshold
        baseline_threshold = rms_noise_value * noise_adjustment_factor
        
        def fit_and_subtract(vector):
            # Identify baseline points
            baseline_mask = np.abs(vector) < baseline_threshold
            baseline_indices = np.where(baseline_mask)[0]

            # Ensure sufficient baseline points
            if len(baseline_indices) < min_baseline_pts:
                # Fallback to first and last points
                baseline_indices = np.concatenate((np.arange(window_size), np.arange(npoints - window_size, npoints)))
                baseline_indices = np.unique(baseline_indices)

            if len(baseline_indices) < order + 1:
                return vector

            # Fit polynomial to baseline points
            x_fit = baseline_indices
            y_fit = vector[x_fit]
            coeffs = np.polyfit(x_fit, y_fit, order)
            baseline = np.polyval(coeffs, np.arange(npoints))

            # Subtract baseline
            corrected_vector = vector - baseline
            return corrected_vector

        # Apply correction along the last axis
        corrected_data = np.apply_along_axis(fit_and_subtract, axis=-1, arr=data)
        
        elapsed = time.perf_counter() - start_time
        corrected_data.processing_history.append(
            {
                'Function': "Time domain polynomial baseline correction",
                'order': order,
                'noise_window_size': noise_window_size,
                'min_baseline_fraction': min_baseline_fraction,
                'noise_adjustment_factor': noise_adjustment_factor,
                'rms_noise_value': rms_noise_value,
                'baseline_threshold': baseline_threshold,
                'time_elapsed_s': elapsed,
                'time_elapsed_str': _format_elapsed_time(elapsed),
            }
        )
        
        return corrected_data
    
    elif domain == "frequency":
        # Handle aliases
        if sx1 is not None: sub_start = sx1
        if sxn is not None: sub_end = sxn
        if fx1 is not None: fit_start = fx1
        if fxn is not None: fit_end = fxn
        if x1 is not None: start = x1
        if xn is not None: end = xn
        if nl is not None: node_list = nl
        if nw is not None: node_width = nw
        if ord is not None: order = ord
        if nc is not None: initial_fit_nodes = nc
        if first is not None: use_first_points = first
        if last is not None: use_last_points = last
        if avg is not None: use_node_avg = avg
        if filt is not None: sine_filter = filt
        
        # Overwrite subtraction region range and fit region range
        if start is not None:
            sub_start = start
            fit_start = start
        if end is not None:
            sub_end = end
            fit_end = end
            
        if sub_end == -1: sub_end = data.shape[-1] - 1
        if fit_end == -1: fit_end = data.shape[-1] - 1
        
        result = data.copy()
        npoints = result.shape[-1]
        
        node_groups = []
        if node_list is not None:
            node_list = [_convert_to_index(result, n, npoints, default=None) for n in node_list]
            for center in node_list:
                if center is None: continue
                
                node_group_start = max(0, center - node_width)
                node_group_end = min(npoints, center + node_width + 1)
                node_group = list(range(node_group_start, node_group_end))
                node_groups.extend(node_group)

        node_groups = sorted(set(node_groups))
        
        sub_start_idx = _convert_to_index(result, sub_start, npoints, default=0)
        sub_end_idx = _convert_to_index(result, sub_end, npoints, default=npoints - 1)
        fit_start_idx = _convert_to_index(result, fit_start, npoints, default=0)
        fit_end_idx = _convert_to_index(result, fit_end, npoints, default=npoints - 1)
        
        
        def fit_and_subtract(vector: np.ndarray) -> np.ndarray:
            if node_groups:
                x_fit = np.array(node_groups)
                y_fit = vector[node_groups]
            else:
                # Default to using the entire fit range
                x_fit = np.arange(fit_start_idx, fit_end_idx + 1)
                y_fit = vector[fit_start_idx:fit_end_idx + 1]
            
            if len(x_fit) < order + 1:
                return vector
            
            # Subtraction range
            x_subtract = np.arange(sub_start_idx, sub_end_idx + 1)
            
            coeffs = np.polyfit(x_fit, y_fit, order)
            baseline = np.polyval(coeffs, x_subtract)
            
            corrected = vector.copy()
            corrected[sub_start_idx:sub_end_idx + 1] -= baseline
            
            return corrected
        
        corrected_data = np.apply_along_axis(lambda v: fit_and_subtract(v), axis=-1, arr=data)
        
        elapsed = time.perf_counter() - start_time
        corrected_data.processing_history.append(
            {
                'Function': "Frequency domain polynomial baseline correction",
                'sub_start': sub_start,
                'sub_end': sub_end,
                'fit_start': fit_start,
                'fit_end': fit_end,
                'order': order,
                'node_list': node_list,
                'node_width': node_width,
                'use_first_points': use_first_points,
                'use_last_points': use_last_points,
                'use_node_avg': use_node_avg,
                'sine_filter': sine_filter,
                'n_nodes': len(node_groups),
                'time_elapsed_s': elapsed,
                'time_elapsed_str': _format_elapsed_time(elapsed),
            }
        )
        
        return corrected_data

# NMRPipe alias
POLY = polynomial_baseline_correction
POLY.__doc__ = polynomial_baseline_correction.__doc__  # Auto-generated
POLY.__name__ = "POLY"  # Auto-generated



def transpose(
    data: NMRData,
    *,
    axes: list[int] = None,
    hyper_complex: bool = False,
    # Aliases
    hyper: bool = False,
) -> NMRData:
    """
    Transpose the data and reorder metadata accordingly.

    Args:
        data (NMRData): The data to transpose.
        axes (list[int], optional): New axis order. If None, reverse axes.
        hyper_complex (bool, optional): Flag to perform hyper complex transpose.
        
    Aliases:
        hyper: Alias for hyper_complex.

    Returns:
        NMRData: Transposed data.
    """
    start_time = time.perf_counter()

    if axes:
        axes = axes[0]
    else:
        axes = list(range(data.ndim))
        axes.reverse()
    
    if hyper:
        raise NotImplementedError
    else:
        result = super(NMRData, data).transpose(*axes)
        

    # Copy attributes
    for attr in data._custom_attrs:
        match attr:
            # Reorder and copy
            case 'axes':
                setattr(result, attr, [getattr(data, attr)[ax] for ax in axes])
            # Deep copy
            case _:
                setattr(result, attr, copy.deepcopy(getattr(data, attr)))
    
    
    is_interleaved = result.axes[-1].get("interleaved_data", None)
    if is_interleaved == True:
        result = _interleaved_to_complex(result)
        result.axes[-1]["interleaved_data"] = False

    # Record processing history
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Transpose",
            'axes': [list(axes) if hasattr(axes, '__iter__') else [axes]],
            'shape_before': data.shape,
            'shape_after': result.shape,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    start_time = time.perf_counter()
    
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn
    
    
    if (
        constant is None
        and constant_real is None
        and constant_imaginary is None
    ):
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


    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Add constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    start_time = time.perf_counter()
    
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn
    
    
    if (
        constant is None
        and constant_real is None
        and constant_imaginary is None
    ):
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


    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Multiply constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
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
    start_time = time.perf_counter()
    
    # Handle aliases
    if r is not None: constant_real = r
    if i is not None: constant_imaginary = i
    if c is not None: constant = c
    if x1 is not None: start = x1
    if xn is not None: end = xn

    if (
        constant is None
        and constant_real is None
        and constant_imaginary is None
    ):
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


    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Set to constant",
            'start': start,
            'end': end,
            'constant': constant,
            'constant_real': constant_real,
            'constant_imaginary': constant_imaginary,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        }
    )

    return result

# NMRPipe alias
SET = set_to_constant
SET.__doc__ = set_to_constant.__doc__  # Auto-generated
SET.__name__ = "SET"  # Auto-generated


def delete_imaginaries(data: NMRData) -> NMRData:
    """
    Discard the imaginary part of complex-valued NMRData.

    Args:
        data (NMRData): Complex NMRData.

    Returns:
        NMRData: Real-valued data.
    """
    start_time = time.perf_counter()

    # Take the real part only
    real_data = np.real(data).copy()

    # Create new NMRData object with real data and preserved metadata
    result = NMRData(real_data, copy_from=data)

    # Record processing history
    elapsed = time.perf_counter() - start_time
    result.processing_history.append({
        'Function': "Delete imaginary part",
        'imag_removed': True,
        'dtype_before': str(data.dtype),
        'dtype_after': str(real_data.dtype),
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return result

# NMRPipe alias
DI = delete_imaginaries
DI.__doc__ = delete_imaginaries.__doc__  # Auto-generated
DI.__name__ = "DI"  # Auto-generated


def null(data: NMRData) -> NMRData:
    """
    Leave data unchanged.

    Args:
        data (NMRData): Input data.

    Returns:
        NMRData: Unchagned NMRData.
    """
    start_time = time.perf_counter()
    
    result = data.copy()
    
    elapsed = time.perf_counter() - start_time
    result.processing_history.append(
        {
            'Function': "Null",
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        }
    )
    return result

# NMRPipe alias
NULL = null
NULL.__doc__ = null.__doc__  # Auto-generated
NULL.__name__ = "NULL"  # Auto-generated


def reverse(
    data: NMRData,
    *,
    adjust_spectral_width: bool = True,
    # Aliases
    sw: bool = None,
) -> NMRData:
    """
    Reverse NMRData in the last dimension.

    Args:
        data (NMRData): Input NMR dataset to reverse.
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after reversing.
        
    Aliases:
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Reversed NMRdata, optionally with updated spectral calibration.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if sw is not None: adjust_spectral_width = sw
    
    reversed_data = data[..., ::-1]

    
    new_data = NMRData(reversed_data, copy_from=data)
    
    if adjust_spectral_width:
        dim = -1 
        full_size = data.shape[dim]
        new_size = new_data.shape[dim]

        # Adjust SW, ORI, and OBS based on ppm limits
        sw, ori, obs = (data.axes[dim][k] for k in ("SW", "ORI", "OBS"))
        
        new_axis_dict = data.axes[dim].copy()
        
        if data.axes[dim]["unit"] == "pts":
            # Calculate the ORI adjustment by one point equivalent
            point_shift = sw / full_size
            new_ori = ori + point_shift # Adjust ORI by one point shift
            new_ppm_scale = new_axis_dict['scale'][::-1] # Reverse the ppm scale
        
        else:
            # Recalculate ppm scale 
            ppm_scale = get_ppm_scale(full_size, sw, ori, obs)
            new_ppm_scale = ppm_scale[::-1]  # Reverse the ppm scale
            ppm_min, ppm_max = new_ppm_scale.min(), new_ppm_scale.max()
            
            new_ori = obs * ppm_max # Adjust ORI to reflect the new ppm max

        new_axis_dict['SW'] = sw # remains unchanged
        new_axis_dict['ORI'] = new_ori
        new_axis_dict['OBS'] = obs # OBS remains unchanged
        new_axis_dict['scale'] = new_ppm_scale
        new_data.axes[dim] = new_axis_dict


    elapsed = time.perf_counter() - start_time
    new_data.processing_history.append({
        'Function': "Reverse data",
        'adjusted_sw': adjust_spectral_width,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return new_data

# NMRPipe alias
REV = reverse
REV.__doc__ = reverse.__doc__  # Auto-generated
REV.__name__ = "REV"  # Auto-generated


def right_shift(
    data: NMRData,
    *,
    shift_amount: int = 0,
    adjust_spectral_width: bool = True,
    # Aliases
    rs: int = None,
    sw: bool = None,
) -> NMRData:
    """
    Apply a right shift and zero pad to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        shift_amount (int): Number of points to shift. Positive for right shift, negative for left shift.
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after shifting.
        
    Aliases:
        rs: Alias for shift_amount.
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Data after applying right shift and zero padding.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if rs is not None: shift_amount = rs
    if sw is not None: adjust_spectral_width = sw
    
    dim = -1 
    npoints = data.shape[dim]
    shift_points = int(np.round(shift_amount))
    
    shift_points = max(min(shift_points, npoints), -npoints)
    
    shifted_data = np.zeros_like(data)
    if shift_points > 0:
        shifted_data[..., shift_points:] = data[..., :-shift_points]
    
    elif shift_points < 0:
        shifted_data[..., :shift_points] = data[..., -shift_points:]
    
    else:
        shifted_data = data.copy()
    
    new_data = NMRData(shifted_data, copy_from=data)
    
    if adjust_spectral_width:
        sw, ori, obs = (data.axes[dim][k] for k in ("SW", "ORI", "OBS"))
        axis_unit = data.axes[dim].get("unit", "pts")
        
        point_shift = sw / npoints

        
        # Adjust ORI based on shift
        if axis_unit == "pts":
            new_ori = ori - (point_shift * shift_points)
        else:
            # Recalculate ppm scale
            ppm_scale = get_ppm_scale(npoints, sw, ori, obs)
            new_ppm_scale = np.roll(ppm_scale, shift_points)
            new_ori = obs * new_ppm_scale[0]

        # Update axis dictionary
        new_axis_dict = data.axes[dim].copy()
        new_axis_dict['SW'] = sw  # SW remains unchanged
        new_axis_dict['ORI'] = new_ori
        new_axis_dict['OBS'] = obs  # OBS remains unchanged
        new_axis_dict['scale'] = np.roll(new_axis_dict['scale'], shift_points)

        new_data.axes[dim] = new_axis_dict


    elapsed = time.perf_counter() - start_time
    new_data.processing_history.append({
        'Function': "Right shift data",
        'shift_amount': shift_amount,
        'adjusted_sw': adjust_spectral_width,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return new_data

# NMRPipe alias
RS = right_shift
RS.__doc__ = right_shift.__doc__  # Auto-generated
RS.__name__ = "RS"  # Auto-generated


def left_shift(
    data: NMRData,
    *,
    shift_amount: int = 0,
    adjust_spectral_width: bool = True,
    # Aliases
    rs: int = None,
    sw: bool = None,
) -> NMRData:
    """
    Apply a left shift and zero pad to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        shift_amount (int): Number of points to shift. Positive for left shift, negative for right shift.
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after shifting.
        
    Aliases:
        rs: Alias for shift_amount.
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Data after applying right shift and zero padding.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if rs is not None: shift_amount = rs
    if sw is not None: adjust_spectral_width = sw
    
    
    new_data = right_shift(data, shift_amount*-1, adjust_spectral_width)
    new_data.processing_history.pop()


    elapsed = time.perf_counter() - start_time
    new_data.processing_history.append({
        'Function': "Left shift data",
        'shift_amount': shift_amount,
        'adjusted_sw': adjust_spectral_width,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return new_data

# NMRPipe alias
LS = left_shift
LS.__doc__ = left_shift.__doc__  # Auto-generated
LS.__name__ = "LS"  # Auto-generated


def circular_shift(
    data: NMRData,
    *,
    right_shift_amount: int = 0,
    left_shift_amount: int = 0,
    negate_shifted: bool = False,
    adjust_spectral_width: bool = True,
    # Aliases
    rs: int = None,
    ls: int = None,
    neg: bool = None,
    sw: bool = None,
) -> NMRData:
    """
    Apply a circular shift to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        right_shift_amount (int): Number of points to right shift.
        left_shift_amount (int): Number of points to left shift.
        negate_shifted (bool): If True, negate the shifted data.
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after shifting.

    Aliases:
        rs: Alias for right_shift_amount.
        ls: Alias for left_shift_amount.
        neg: Alias for negate_shifted.
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Data after applying circular shift.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if rs is not None: right_shift_amount = rs
    if ls is not None: left_shift_amount = ls
    if neg is not None: negate_shifted = neg
    if sw is not None: adjust_spectral_width = sw
    
    if right_shift_amount != 0 and left_shift_amount != 0:
        raise ValueError("Specify only one of right_shift_amount or left_shift_amount, not both.")
    
    
    shift_amount = right_shift_amount if right_shift_amount != 0 else -left_shift_amount
    
    dim = -1
    npoints = data.shape[dim]
    shift_amount = shift_amount % npoints
    
    shifted_data = np.roll(data, shift_amount, axis=dim)
    
    if negate_shifted:
        if shift_amount > 0:
            shifted_data[..., :shift_amount] *= -1
        elif shift_amount < 0:
            shifted_data[..., shift_amount:] *= -1
    
    new_data = NMRData(shifted_data, copy_from=data)


    if adjust_spectral_width:
        # Extract SW, ORI, OBS
        sw_value, ori, obs = (data.axes[dim][k] for k in ("SW", "ORI", "OBS"))

        # Calculate the point shift adjustment
        point_shift = sw_value / npoints

        # Adjust ORI based on shift
        new_ori = ori - (point_shift * shift_amount)

        # Update axis dictionary
        new_axis_dict = data.axes[dim].copy()
        new_axis_dict['SW'] = sw_value  # SW remains unchanged
        new_axis_dict['ORI'] = new_ori
        new_axis_dict['OBS'] = obs  # OBS remains unchanged
        new_axis_dict['scale'] = np.roll(new_axis_dict['scale'], shift_amount)

        new_data.axes[dim] = new_axis_dict


    elapsed = time.perf_counter() - start_time
    new_data.processing_history.append({
        'Function': "Circular shift data",
        'shift_amount': shift_amount,
        'negate_shifted': negate_shifted,
        'adjusted_sw': adjust_spectral_width,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return new_data

# NMRPipe alias
CS = circular_shift
CS.__doc__ = circular_shift.__doc__  # Auto-generated
CS.__name__ = "CS"  # Auto-generated


def manipulate_sign(
    data: NMRData,
    *,
    negate_all: bool = False,
    negate_reals: bool = False,
    negate_imaginaries: bool = False,
    negate_left_half: bool = False,
    negate_right_half: bool = False,
    alternate_sign: bool = False,
    absolute_value: bool = False,
    replace_with_sign: bool = False,
    # Aliases
    ri: bool = None,
    r: bool = None,
    i: bool = None,
    left: bool = None,
    right: bool = None,
    alt: bool = None,
    abs: bool = None,
    sign: bool = None,
) -> NMRData:
    """
    Apply various sign manipulations to the NMR data.

    Args:
        data (NMRData): Input NMR dataset.
        negate_all (bool): Negate the entire dataset.
        negate_reals (bool): Negate only the real part of the data.
        negate_imaginaries (bool): Negate only the imaginary part of the data.
        negate_left_half (bool): Negate the left half of the data.
        negate_right_half (bool): Negate the right half of the data.
        alternate_sign (bool): Alternate sign for each point.
        absolute_value (bool): Apply absolute value to the entire dataset.
        replace_with_sign (bool): Replace each value with its sign (+1, 0, -1).

    Aliases:
        ri: Alias for negate_all.
        r: Alias for negate_reals.
        i: Alias for negate_imaginaries.
        left: Alias for negate_left_half.
        right: Alias for negate_right_half.
        alt: Alias for alternate_sign.
        abs: Alias for absolute_value.
        sign: Alias for replace_with_sign.

    Returns:
        NMRData: Data after sign manipulation.
    """
    start_time = time.perf_counter()
    
    # Handle aliases
    if ri is not None: negate_all = ri
    if r is not None: negate_reals = r
    if i is not None: negate_imaginaries = i
    if left is not None: negate_left_half = left
    if right is not None: negate_right_half = right
    if alt is not None: alternate_sign = alt
    if abs is not None: absolute_value = abs
    if sign is not None: replace_with_sign = sign
    
    result = data.copy()
    npoints = result.shape[-1]

    if negate_all:
        result *= -1

    if negate_reals:
        result.real *= -1

    if negate_imaginaries:
        result.imag *= -1

    if negate_left_half:
        midpoint = npoints // 2
        result[..., :midpoint] *= -1

    if negate_right_half:
        midpoint = npoints // 2
        result[..., midpoint:] *= -1

    if alternate_sign:
        sign_pattern = np.ones(npoints)
        sign_pattern[1::2] = -1 
        result *= sign_pattern

    if absolute_value:
        result = np.abs(result)

    # Replace each value with its sign (+1, 0, -1)
    if replace_with_sign:
        result = np.sign(result)

    elapsed = time.perf_counter() - start_time
    result.processing_history.append({
        'Function': "Sign manipulation",
        'negate_all': negate_all,
        'negate_reals': negate_reals,
        'negate_imaginaries': negate_imaginaries,
        'negate_left_half': negate_left_half,
        'negate_right_half': negate_right_half,
        'alternate_sign': alternate_sign,
        'absolute_value': absolute_value,
        'replace_with_sign': replace_with_sign,
        'time_elapsed_s': elapsed,
        'time_elapsed_str': _format_elapsed_time(elapsed),
    })

    return result

# NMRPipe alias
SIGN = manipulate_sign
SIGN.__doc__ = manipulate_sign.__doc__  # Auto-generated
SIGN.__name__ = "SIGN"  # Auto-generated