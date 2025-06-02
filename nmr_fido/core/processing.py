from time import perf_counter
import numpy as np
import copy
from nmr_fido.nmrdata import NMRData
from nmr_fido.utils.scales import get_ppm_scale
from nmr_fido.utils.unit_to_index import _convert_to_index
from scipy.signal import hilbert
from scipy import signal, odr
from scipy.optimize import curve_fit
from typing import TypeVar, cast


NMRArrayType = TypeVar("NMRArrayType", bound=np.ndarray)


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


def _interleaved_to_complex(data: NMRArrayType, dim: int = -1) -> NMRArrayType:
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
    
    result = complex_data.copy()
    
    if isinstance(data, NMRData):
        result = NMRData(complex_data, copy_from=data)
        result.axes[dim]["acqu_mode"] = "Complex"
        return result

    return complex_data.view(type(data))


def _lowpass_filter_safe(fid: NMRArrayType, filt: np.ndarray) -> NMRArrayType:
    # Half filter width
    K = len(filt) // 2

    # Pad signal edges by reflection to avoid wrap-around artifacts
    padded = np.pad(fid, (K, K), mode='reflect')

    # Convolve with filter on padded data, 'valid' mode returns filtered signal matching original length
    conv = signal.convolve(padded, filt, mode='valid')

    # Normalize by sum of filter coefficients to preserve amplitude scale
    conv /= filt.sum()

    return conv


def solvent_filter(
    data: NMRArrayType,
    *,
    filter_mode: int | str = "Low Pass",
    lowpass_size: int = 16,
    lowpass_shape: int | str = "Boxcar",
    butter_ord: int = 4,
    butter_cutoff: float = 0.05,
    poly_ext_order: int = 2,
    spline_noise: float = 1.0,
    smooth_factor: float = 1.1,
    skip_points: int = 0,
    use_poly_ext: bool = True,
    use_mirror_ext: bool = False,
    
    # Aliases
    mode: int | None = None,
    fl: int | None = None,
    fs: int | None = None,
    po: int | None = None,
    sn: float | None = None,
    sf: float | None = None,
    head: int | None = None,
    poly: bool | None = None,
    mir: bool | None = None,
    #noseq: bool | None = None,
    #nodms: bool | None = None,
) -> NMRArrayType:
    """
    Desc.

    Args:
        data (NMRArrayType): Input data.

    Aliases:

    Returns:
        NMRArrayType: .
    """
    start_time = perf_counter()
    
    filter_mode = mode if mode is not None else filter_mode
    lowpass_size = fl if fl is not None else lowpass_size
    lowpass_shape = fs if fs is not None else lowpass_shape
    poly_ext_order = po if po is not None else poly_ext_order
    spline_noise = sn if sn is not None else spline_noise
    smooth_factor = sf if sf is not None else smooth_factor
    skip_points = head if head is not None else skip_points
    use_poly_ext = poly if poly is not None else use_poly_ext
    use_mirror_ext = mir if mir is not None else use_mirror_ext

    
    if isinstance(filter_mode, int):
        filter_mode = {1: "Low Pass", 2: "Spline", 3: "Polynomial"}[filter_mode]

    if isinstance(lowpass_shape, int):
        lowpass_shape = {1: "Boxcar", 2: "Sine", 3: "Sine^2"}[lowpass_shape]


    result = data.copy()
    sliced_data = result[..., skip_points:]
    
    filter_width = lowpass_size*2 + 1

    match filter_mode:
        case "Low Pass":
            """
            More info on how these work:
            
            Overview:
            Cross 1996 -> DOI: https://doi.org/10.1016/S0922-3487(96)80043-8
            
            Gaussian filter:
            Marion et al. 1989 -> DOI: https://doi.org/10.1016/0022-2364(89)90391-0
            
            From Marion et al.:
                "The residual H20 signal is responsible for the low-frequency component
                of the signal. To a good approximation, this low-frequency component of the FID
                can be calculated by averaging neighboring time-domain data points, which is equiv-
                alent to convolution with a rectangular function. The width of the rectangle corre-
                sponds to the number of time-domain data points that are averaged. This low-fre-
                quency component is then subtracted from the original signal (Fig. 1B)."
            """
            filter_kernel = None
            b = a = None  # Predeclare for Butterworth
            match lowpass_shape:
                case "Boxcar":
                    filter_kernel = np.ones(filter_width, dtype=np.float32)
                
                case "Sine":
                    filter_kernel = np.cos(np.pi * np.linspace(-0.5, 0.5, filter_width))
                
                case "Sine^2":
                    filter_kernel = np.cos(np.pi * np.linspace(-0.5, 0.5, filter_width)) ** 2
                    
                case "Gaussian":
                    filter_kernel = np.exp(-4 * (np.linspace(-0.5, 0.5, filter_width)**2) / (0.5**2))
                    
                case "Butterworth":
                    b, a = signal.butter(butter_ord, butter_cutoff, btype='low', analog=False) # type: ignore
                    
                case _:
                    raise ValueError(f"Unknown lowpass_shape: {lowpass_shape}")
            
            if filter_kernel is not None:
                # FIR filter via convolution with safe padding
                for index in np.ndindex(sliced_data.shape[:-1]):
                    fid = sliced_data[index]
                    filtered_fid = _lowpass_filter_safe(fid, filter_kernel)
                    sliced_data[index] = fid - filtered_fid

            elif lowpass_shape == "Butterworth":
                # IIR Butterworth filter applied forwards and backwards
                for index in np.ndindex(sliced_data.shape[:-1]):
                    fid = sliced_data[index]
                    filtered_fid = signal.filtfilt(b, a, fid)
                    sliced_data[index] = fid - filtered_fid

            else:
                pass

        
        case "Spline":
            raise NotImplementedError("Spline filter mode not implemented yet.")
        
        case "Polynomial":
            """
            Possible implementation:
            
            Bielecki and Levitt 1989 -> https://doi.org/10.1016/0022-2364(89)90218-7
            """
            raise NotImplementedError("Polynomial filter mode not implemented yet.")
        
        
        case _:
            raise ValueError(f"Unknown filter mode: {filter_mode}")


    result[..., skip_points:] = sliced_data
    
    elapsed = perf_counter() - start_time
    if isinstance(result, NMRData):
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


def _fit_lp_coeff(vector: np.ndarray, pred_start_idx: int, pred_end_idx: int, order: int) -> np.ndarray:
    """
           m
    x_k =  Σ q_i * s_k-i
          i=1
    """
    
    y_fit = vector[pred_start_idx:pred_end_idx + 1]
    
    K = order
    n = len(y_fit) - order
    
    """
    q1*x0      + q2*x1  + q3*x2      + ... + qK*x_{K-1}   = x_K
    q1*x1      + q2*x2  + q3*x3      + ... + qK*x_K       = x_{K+1}
    q1*x2      + q2*x3  + q3*x4      + ... + qK*x_{K+1}   = x_{K+2}
    
        ⋮
    
    q1*x_{n-1} + q2*x_n + q3*x_{n+1} + ... + qK*x_{n+K-2} = x_{n+K-1}
    """
    M = np.array([y_fit[i : i + K] for i in range(n)])
    
    r = y_fit[K : K + n]
    
    coeffs, residuals, rank, s = np.linalg.lstsq(M, r, rcond=None)
    
    return coeffs


def _find_roots(coeffs: np.ndarray) -> np.ndarray:
    char_poly = np.concatenate(([1.0+0.0j], -coeffs[::-1]))
    return np.roots(char_poly)


def _fix_roots(roots: np.ndarray, root_fix_mode: str) -> np.ndarray:
    match root_fix_mode:
        case "suppress_increasing":
            # Reflect roots outside unit circle
            return np.array([
                1 / np.conj(r) if abs(r) > 1 else r
                for r in roots
            ])
        
        case "suppress_decreasing":
            # Reflect roots inside unit circle
            return np.array([
                1 / np.conj(r) if abs(r) < 1 else r
                for r in roots
            ])
            
        case _:
            return roots


def _plot_roots(*roots_list: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k', label='Unit Circle')
    ax.axhline(0, color="black", zorder=0)
    ax.axvline(0, color="black", zorder=0)
    for i, roots in enumerate(roots_list):
        ax.scatter(roots.real, roots.imag, ec="k", zorder=9999)
    ax.set_aspect('equal')
    ax.grid(True)
    fig.show()
    return


def _plot_char_poly(*coeffs_list: np.ndarray) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    theta = np.linspace(0, 2 * np.pi, 800)
    z = np.exp(1j * theta)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(cmap.N)]

    for i, coeffs in enumerate(coeffs_list):
        full_poly = np.concatenate(([1.0], -coeffs))
        values = np.polyval(full_poly, z)
        color = colors[i % len(colors)]
        ax.plot(theta, 20 * np.log10(np.abs(values)), label=f'Poly {i+1}', color=color)

    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Characteristic Polynomial Magnitudes on Unit Circle")
    ax.grid(True)
    ax.legend()
    fig.show()


def linear_prediction(
    data: NMRArrayType,
    *,
    prediction_size: int = -1,
    pred_start: int = 0,
    pred_end: int = -1,
    order: int = 8,
    model_direction: str = "forward",
    prediction_direction: str = "forward",
    fix_roots: bool = True,
    root_fix_mode: str = "auto",
    mirror_image: bool = False,
    shifted_mirror_image: bool = False,
    # Aliases
    pred: int | None = None,
    x1: int | None = None,
    xn: int | None = None,
    ord: int | None = None,
    f: bool | None = None,
    b: bool | None = None,
    fb: bool | None = None,
    before: bool | None = None,
    after: bool | None = None,
    nofix: bool | None = None,
    fix: bool | None = None,
    fixMode: int | None = None,
    ps90_180: bool | None = None,
    ps0_0: bool | None = None,
    #pca: bool | None = None,
    #extra: int | None = None,
) -> NMRArrayType:
    """
    Apply linear prediction to the last dimension of the NMRData array.

    Args:
        data (NMRData): Input data.
        prediction_size (int): Number of points to predict (default: same as data size).
        pred_start (int): Start index for fitting.
        pred_end (int): End index for fitting.
        order (int): Number of coefficients to fit.
        direction (str): 'forward', 'backward', or 'both' (currently only 'forward' implemented).
        use_root_fixing (bool): Whether to apply root-fixing to suppress diverging behavior.
        root_fix_mode (str): Strategy to suppress diverging roots.

    Returns:
        NMRData: Predicted data with extended FID.
    """
    start_time = perf_counter()
    
    # Handle argument aliases
    prediction_size = pred if pred is not None else prediction_size
    pred_start = x1 if x1 is not None else pred_start
    pred_end = xn if xn is not None else pred_end
    order = ord if ord is not None else order
    mirror_image = ps90_180 if ps90_180 is not None else mirror_image
    shifted_mirror_image = ps0_0 if ps0_0 is not None else shifted_mirror_image

    
    if f: model_direction = "forward"
    if b: model_direction = "backward"
    if fb: model_direction = "both"

    if before is not None:
        prediction_direction = "backward"
    if after is not None:
        prediction_direction = "forward"

    if nofix:
        fix_roots = False
    if fix:
        fix_roots = True

    if fixMode is not None:
        root_fix_mode = {-1: "suppress_decreasing", 0: None, 1: "suppress_increasing"}[fixMode]
    
    
    result = data.copy()
    npoints = result.shape[-1]
    
    if prediction_size == -1: prediction_size = npoints
    
    if root_fix_mode == "auto":
        root_fix_mode = {"forward": "suppress_increasing", "backward": "suppress_decreasing"}[prediction_direction]
    
    
    if order >= npoints/2:
        raise ValueError(f"Number of coefficients ({order=}) must be less than half the number of points in the vector ({npoints=})")
    
    
    pred_start_idx = _convert_to_index(result, pred_start, npoints, default=0)
    pred_end_idx = _convert_to_index(result, pred_end, npoints, default=npoints - 1)
    
    
    # Predict points
    original_shape = result.shape
    new_last_dim = original_shape[-1] + prediction_size
    new_shape = original_shape[:-1] + (new_last_dim,)
    predicted_data = np.zeros(new_shape)
    
    
    predict_reverse = prediction_direction == "backward"
    for index in np.ndindex(result.shape[:-1]):
        fid = np.array(result[index])
        fid_length = fid.size
        
        coeffs = np.zeros(order)
        model_fid = fid
        
        if mirror_image:
            mirror = np.conj(fid[::-1])
            model_fid = np.concatenate([mirror[:-1], fid])
            
        if shifted_mirror_image:
            mirror = np.conj(fid[::-1])
            model_fid = np.concatenate([mirror, fid])
        
        match model_direction:
            case "forward":
                coeffs = _fit_lp_coeff(model_fid, pred_start_idx, pred_end_idx, order)
                
            case "backward":
                coeffs = _fit_lp_coeff(model_fid[::-1], pred_start_idx, pred_end_idx, order)
                
            case "both":
                coeff_fwd = _fit_lp_coeff(model_fid, pred_start_idx, pred_end_idx, order)
                coeff_rev = _fit_lp_coeff(model_fid[::-1], pred_start_idx, pred_end_idx, order)
                coeffs = 0.5 * (coeff_fwd + coeff_rev)
        
        
        if fix_roots:
            roots = _find_roots(coeffs)
            
            fixed_roots = _fix_roots(roots, root_fix_mode)
            
            fixed_coeffs = -np.poly(fixed_roots)[:0:-1] # drop leading 1 and flip sign convention
            coeffs = fixed_coeffs 
        

        extended = np.empty(fid_length + prediction_size, dtype=fid.dtype)
        extended[:fid_length] = fid[::-1] if predict_reverse else fid

        for i in range(prediction_size):
            prev_points = extended[i + fid_length - order : i + fid_length]
            extended[fid_length + i] = np.dot(coeffs, prev_points)

        predicted_fid = extended[::-1] if predict_reverse else extended
        predicted_data[index] = predicted_fid
    
    
    if isinstance(data, NMRData):
        result = NMRData(predicted_data, copy_from=data)
        result.axes[-1]["scale"] = np.arange(new_last_dim)
        
        elapsed = perf_counter() - start_time
        result.processing_history.append({
            'Function': "Linear Prediction",
            'order': order,
            'prediction_size': prediction_size,
            'pred_start_idx': pred_start_idx,
            'pred_end_idx': pred_end_idx,
            'model_direction': model_direction,
            'prediction_direction': prediction_direction,
            'fix_roots': fix_roots,
            'root_fix_mode': root_fix_mode,
            'shape_before': original_shape,
            'shape_after': predicted_data.shape,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })
    
    return result

# NMRPipe alias
LP = linear_prediction
LP.__doc__ = linear_prediction.__doc__  # Auto-generated
LP.__name__ = "LP"  # Auto-generated


def _apply_window(
    data: NMRArrayType,
    window: np.ndarray,
    size_window: int,
    start: int,
    invert_window: bool,
    scale_factor_first_point: float,
    fill_outside_one: bool,
) -> NMRArrayType:
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
    
    if isinstance(data, NMRData):
        return NMRData(result, copy_from=data)
    
    
    return result.view(type(data))


def sine_bell_window(
    data: NMRArrayType,
    *,
    start_angle: float = 0.0,
    end_angle: float = 1.0,
    exponent: float = 1.0,
    size_window: int | None = None,
    start: int = 1,
    scale_factor_first_point: float = 1.0,
    fill_outside_one: bool = False,
    invert_window: bool = False,
    # Aliases
    off: float | None = None,
    end: float | None = None,
    pow: float | None = None,
    size: int | None = None,
    c: float | None = None,
    one: bool | None = None,
    inv: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle argument aliases
    start_angle = off if off is not None else start_angle
    end_angle = end if end is not None else end_angle
    exponent = pow if pow is not None else exponent
    size_window = size if size is not None else size_window
    scale_factor_first_point = c if c is not None else scale_factor_first_point
    fill_outside_one = one if one is not None else fill_outside_one
    invert_window = inv if inv is not None else invert_window

    
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
    
    result = _apply_window(
        data, window,
        size_window,
        start,
        invert_window,
        scale_factor_first_point,
        fill_outside_one
    )
    
    elapsed = perf_counter() - start_time
    if isinstance(result, NMRData) and hasattr(result, "processing_history"):
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
    data: NMRArrayType,
    *,
    inv_exp_width: float = 0.0,
    broaden_width: float = 0.0,
    center: float = 0.0,
    size_window: int | None = None,
    start: int = 1,
    scale_factor_first_point: float = 1.0,
    fill_outside_one: bool = False,
    invert_window: bool = False,
    sw: float | None = None,
    # Aliases
    g1: float | None = None,
    g2: float | None = None,
    g3: float | None = None,
    size: int | None = None,
    c: float | None = None,
    one: bool | None = None,
    inv: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle argument aliases
    inv_exp_width = g1 if g1 is not None else inv_exp_width
    broaden_width = g2 if g2 is not None else broaden_width
    center = g3 if g3 is not None else center
    size_window = size if size is not None else size_window
    scale_factor_first_point = c if c is not None else scale_factor_first_point
    fill_outside_one = one if one is not None else fill_outside_one
    invert_window = inv if inv is not None else invert_window

    
    if size_window is None:
        size_window = int(data.shape[-1])
    
    if isinstance(data, NMRData):
        sw = data.axes[-1].get("SW", None)
        if sw is None:
            raise ValueError("Spectral width (SW) is not defined in the data axis.")
    elif sw is None:
        raise ValueError("Spectral width (sw) must be provided when data is not NMRData.")
    
    # Create window
    t = np.arange(size_window)
    npoints = data.shape[-1]
    center_index = int(center * (npoints - 1))
    
    exp_component = np.exp((np.pi * t * inv_exp_width) / sw)
    gauss_component = np.exp(
        -((0.6 * np.pi * broaden_width * (center_index - t)) ** 2)
    )
    window = (exp_component * gauss_component).astype(data.dtype)
    
    result = _apply_window(
        data, window,
        size_window,
        start,
        invert_window,
        scale_factor_first_point,
        fill_outside_one,
    )
    
    
    elapsed = perf_counter() - start_time
    if isinstance(result, NMRData):
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
            'SW': sw,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })
    
    return result

# NMRPipe alias
GM = lorentz_to_gauss_window
GM.__doc__ = lorentz_to_gauss_window.__doc__  # Auto-generated
GM.__name__ = "GM"  # Auto-generated


def exp_mult_window(
    data: NMRArrayType,
    *,
    line_broadening: float = 0.0,
    size_window: int | None = None,
    start: int = 1,
    scale_factor_first_point: float = 1.0,
    fill_outside_one: bool = False,
    invert_window: bool = False,
    sw: float | None = None,
    # Aliases
    lb: float | None = None,
    size: int | None = None,
    c: float | None = None,
    one: bool | None = None,
    inv: bool | None = None,
) -> NMRArrayType:
    """
    Apply an exponential multiply apodization (window) to the last dimension of the data.

    Args:
        data (NMRData): Input data.
        line_broadening (float): Line broadening factor (default 0.0).
        size_window (int, optional): Number of points in the window (default: size of last axis).
        start (int): Index to start applying the window (default 1 = first point).
        scale_factor_first_point (float): Scaling for the first point (default 1.0).
        fill_outside_one (bool): If True, data outside window is multiplied by 1.0 instead of 0.0.
        invert_window (bool): If True, apply 1/window instead of window and 1/scale_factor_first_point.

    Aliases:
        lb: Alias for line_broadening
        size: Alias for size_window
        c: Alias for scale_factor_first_point
        one: Alias for fill_outside_one
        inv: Alias for invert_window

    Returns:
        NMRData: Data after applying exponential multiply apodization.
    """
    start_time = perf_counter()
    
    # Handle argument aliases
    line_broadening = lb if lb is not None else line_broadening
    size_window = size if size is not None else size_window
    scale_factor_first_point = c if c is not None else scale_factor_first_point
    fill_outside_one = one if one is not None else fill_outside_one
    invert_window = inv if inv is not None else invert_window



    
    if size_window is None:
        size_window = int(data.shape[-1])
    
    if isinstance(data, NMRData):
        sw = data.axes[-1].get("SW", None)
        if sw is None:
            raise ValueError("Spectral width (SW) is not defined in the data axis.")
    elif sw is None:
        raise ValueError("Spectral width (sw) must be provided when data is not NMRData.")
    
    # Create window
    t = np.arange(size_window)
    
    window = np.exp(
        -np.pi * t * line_broadening / sw
    ).astype(data.dtype)
    
    result = _apply_window(
        data, window,
        size_window,
        start,
        invert_window,
        scale_factor_first_point,
        fill_outside_one
    )
    
    if isinstance(result, NMRData):
        elapsed = perf_counter() - start_time
        result.processing_history.append({
            'Function': "Apodization: Exponential multiply window",
            'line_broadening': line_broadening,
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
EM = exp_mult_window
EM.__doc__ = exp_mult_window.__doc__  # Auto-generated
EM.__name__ = "EM"  # Auto-generated



def zero_fill(
    data: NMRArrayType,
    *,
    factor: int = 1,
    add: int | None = None,
    final_size: int | None = None,
    # Aliases
    zf: int | None = None,
    pad: int | None = None,
    size: int | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle argument aliases
    factor = zf if zf is not None else factor
    add = pad if pad is not None else add
    final_size = size if size is not None else final_size

    
    original_shape = list(data.shape)
    last_dim = original_shape[-1]
    
    if isinstance(data, NMRData):
        last_unit = data.axes[-1]["unit"]
        if last_unit not in ("pts", None, "points"):
            raise ValueError(
                f"Cannot zero-fill: last dimension unit is '{last_unit}', expected 'pts' or None."
            )

    # If user sets anything other than factor, we switch mode
    if any(x is not None for x in (add, final_size)):
        if sum(x is not None for x in (add, final_size)) > 1:
            raise ValueError("Specify only one of 'add' or 'final_size'.")
        factor = -1  # Ignore default doubling if add or final_size is given

    method = ""
    new_last_dim = last_dim
    if factor != -1:
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

    if isinstance(data, NMRData):
        result = NMRData(result_array, copy_from=data)
        
        # Update last scale with pts
        result.axes[-1]["scale"] = np.arange(new_last_dim)
        
        # Update processing history
        elapsed = perf_counter() - start_time
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
        
    return result_array.view(type(data))

# NMRPipe alias
ZF = zero_fill
ZF.__doc__ = zero_fill.__doc__  # Auto-generated
ZF.__name__ = "ZF"  # Auto-generated



def fourier_transform(
    data: NMRArrayType,
    *,
    real_only: bool = False,
    inverse: bool = False,
    negate_imaginaries: bool = False,
    sign_alteration: bool = False,
    bruk: bool = False,
    #dmx: bool = False,
    #nodmx: bool = False,
    # Aliases
    real: bool | None = None,
    inv: bool | None = None,
    neg: bool | None = None,
    alt: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle argument aliases
    real_only = real if real is not None else real_only
    inverse = inv if inv is not None else inverse
    negate_imaginaries = neg if neg is not None else negate_imaginaries
    sign_alteration = alt if alt is not None else sign_alteration



    
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


    if isinstance(data, NMRData):
        # Result
        result = NMRData(transformed, copy_from=data)
        
        # Convert scale to ppm
        result.scale_to_ppm()

        # Update metadata
        elapsed = perf_counter() - start_time
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

    return transformed.view(type(data))

# NMRPipe alias
FT = fourier_transform
FT.__doc__ = fourier_transform.__doc__  # Auto-generated
FT.__name__ = "FT"  # Auto-generated



def hilbert_transform(
    data: NMRArrayType,
    *,
    mirror_image: bool = False,
    temporary_zero_fill: bool = False,
    size_time_domain: int | None = None,
    # Aliases
    ps90_180: bool | None = None,
    zf: bool | None = None,
    td: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
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
        hilbert_data = np.asarray(hilbert(mirrored, axis=-1))
        result_array = hilbert_data[..., :array.shape[-1]]
    else:
        # Standard Hilbert transform
        result_array = np.asarray(hilbert(array, axis=-1))


    # If temporary zero-filled, crop back
    if temporary_zero_fill:
        result_array = result_array[..., :original_shape[-1]]

    if isinstance(data, NMRData):
        result = NMRData(result_array, copy_from=data)
        
        # Update metadata
        elapsed = perf_counter() - start_time
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
    else:
        result = result_array

    return cast(NMRArrayType, result)

# NMRPipe alias
HT = hilbert_transform
HT.__doc__ = hilbert_transform.__doc__  # Auto-generated
HT.__name__ = "HT"  # Auto-generated



def phase(
    data: NMRArrayType,
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
    inv: bool | None = None,
    ht: bool | None = None,
    zf: bool | None = None,
    exp: bool | None = None,
    tc: float | None = None,
    #rs: bool | None = None,
    #ls: bool | None = None,
    #sw: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle argument aliases
    invert = inv if inv is not None else invert
    reconstruct_imaginaries = ht if ht is not None else reconstruct_imaginaries
    temporary_zero_fill = zf if zf is not None else temporary_zero_fill
    exponential_correction = exp if exp is not None else exponential_correction
    decay_constant = tc if tc is not None else decay_constant

    
    
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
        phase_array = np.deg2rad(p0 * np.exp(-decay_constant * x / npoints))
    else:
        phase_array = np.deg2rad(p0 + p1*( x / npoints))
    
    if invert:
        phase_array = -phase_array
    
    phase_correction = np.exp(1j * phase_array)
    
    result = np.asarray(array) * phase_correction
    
    if temporary_zero_fill:
        result = result[..., :original_shape[-1]]
    
    
    if isinstance(data, NMRData):
        result = NMRData(result, copy_from=data)
        result.processing_history.append({
            'Function': "Phase Correction",
            'p0': p0,
            'p1': p1,
            'invert': invert,
            'exponential_correction': exponential_correction,
            'decay_constant': decay_constant,
            'reconstruct_imaginaries': reconstruct_imaginaries,
            'temporary_zero_fill': temporary_zero_fill,
            'time_elapsed_s': perf_counter() - start_time,
            'time_elapsed_str': _format_elapsed_time(perf_counter() - start_time),
        })
    
    return result.view(type(data))

# NMRPipe alias
PS = phase
PS.__doc__ = phase.__doc__  # Auto-generated
PS.__name__ = "PS"  # Auto-generated



def extract_region(
    data: NMRArrayType,
    *,
    start: str | int | None = None,
    end: str | int | None = None,
    start_y: str | int | None = None,
    end_y: str | int | None = None,
    left_half: bool = False,
    right_half: bool = False,
    middle_half: bool = False,
    power_of_two: bool = False,
    adjust_spectral_width: bool = True,
    multiple_of: int | None = None,
    # Aliases
    #time: bool = None,
    left: bool | None = None,
    right: bool | None = None,
    mid: bool | None = None,
    pow2: bool | None = None,
    sw: bool | None = None,
    round: int | None = None,
    x1: str | None = None,
    xn: str | None = None,
    y1: str | None = None,
    yn: str | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    start = x1 if x1 is not None else start
    end = xn if xn is not None else end
    start_y = y1 if y1 is not None else start_y
    end_y = yn if yn is not None else end_y
    left_half = left if left is not None else left_half
    right_half = right if right is not None else right_half
    middle_half = mid if mid is not None else middle_half
    power_of_two = pow2 if pow2 is not None else power_of_two
    adjust_spectral_width = sw if sw is not None else adjust_spectral_width
    multiple_of = round if round is not None else multiple_of
    
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
    
    # Adjust size if rounding requested
    if power_of_two:
        new_x_size = 2 ** int(np.floor(np.log2(x_size)))
        end_idx = start_idx + new_x_size - 1
    elif multiple_of:
        new_x_size = x_size - (x_size % multiple_of)
        end_idx = start_idx + new_x_size - 1
        
    # Slice the data
    
    
    if adjust_spectral_width:
        slicer = (slice(start_y_idx, end_y_idx + 1), slice(start_idx, end_idx + 1)) if result.ndim > 1 else slice(start_idx, end_idx + 1)
        new_data = result[slicer]
    else:
        # Manual slicing on array level
        array = np.asarray(result)
        sliced = array[start_y_idx:end_y_idx+1, start_idx:end_idx+1] if result.ndim > 1 else array[start_idx:end_idx+1]
        
        new_data = NMRData(sliced, copy_from=result) if isinstance(result, NMRData) else sliced.copy()

    if isinstance(new_data, NMRData):
        new_data.processing_history.append({
            'Function': "Extract Region",
            'start_x': start_idx,
            'end_x': end_idx,
            'start_y': start_y_idx,
            'end_y': end_y_idx,
            'shape_before': result.shape,
            'shape_after': new_data.shape,
            'adjusted_sw': adjust_spectral_width,
            'time_elapsed_s': perf_counter() - start_time,
            'time_elapsed_str': _format_elapsed_time(perf_counter() - start_time),
        })

    return new_data.view(type(data))

# NMRPipe alias
EXT = extract_region
EXT.__doc__ = extract_region.__doc__  # Auto-generated
EXT.__name__ = "EXT"  # Auto-generated


def _pbc_time(
    data: NMRArrayType,
    *,
    order: int,
    noise_window_size: int,
    min_baseline_fraction: float,
    noise_adjustment_factor: float,
    rms_noise_value: float,
) -> NMRArrayType:
    start_time = perf_counter()

    npoints = data.shape[-1]
    window_size = noise_window_size
    min_baseline_pts = int(min_baseline_fraction * npoints)

    if rms_noise_value == 0.0:
        windows = [data[..., i:i + window_size] for i in range(0, npoints, window_size)]
        noise_estimates = [np.std(w, axis=-1) for w in windows if w.shape[-1] == window_size]
        if noise_estimates:
            rms_noise_value = np.median(np.concatenate(noise_estimates), axis=-1)

    baseline_threshold = rms_noise_value * noise_adjustment_factor

    def fit_and_subtract_time(vector: np.ndarray) -> np.ndarray:
        baseline_mask = np.abs(vector) < baseline_threshold
        baseline_indices = np.where(baseline_mask)[0]

        if len(baseline_indices) < min_baseline_pts:
            baseline_indices = np.concatenate((np.arange(window_size), np.arange(npoints - window_size, npoints)))
            baseline_indices = np.unique(baseline_indices)

        if len(baseline_indices) < order + 1:
            return vector

        x_fit = baseline_indices
        y_fit = vector[x_fit]
        coeffs = np.polyfit(x_fit, y_fit, order)
        baseline = np.polyval(coeffs, np.arange(npoints))
        return vector - baseline

    corrected_data = np.apply_along_axis(fit_and_subtract_time, axis=-1, arr=data)

    if isinstance(corrected_data, NMRData):
        elapsed = perf_counter() - start_time
        corrected_data.processing_history.append({
            'Function': "Time domain polynomial baseline correction",
            'order': order,
            'noise_window_size': noise_window_size,
            'min_baseline_fraction': min_baseline_fraction,
            'noise_adjustment_factor': noise_adjustment_factor,
            'rms_noise_value': rms_noise_value,
            'baseline_threshold': baseline_threshold,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })

    return cast(NMRArrayType, corrected_data)


def _pbc_freq(
    data: NMRArrayType,
    *,
    sub_start: int,
    sub_end: int,
    fit_start: int,
    fit_end: int,
    node_list: list[str | int | None] | None,
    node_width: int,
    order: int,
    use_first_points: bool,
    use_last_points: bool,
    use_node_avg: bool,
    sine_filter: bool,
) -> NMRArrayType:
    start_time = perf_counter()

    result = data.copy()
    npoints = result.shape[-1]

    node_groups = []
    if node_list is not None:
        resolved_nodes = [_convert_to_index(result, n, npoints, default=-1) for n in node_list or []]
        for center in resolved_nodes:
            if center == -1: continue
            
            node_group_start = max(0, center - node_width)
            node_group_end = min(npoints, center + node_width + 1)
            node_group = list(range(node_group_start, node_group_end))
            node_groups.extend(node_group)

    node_groups = sorted(set(node_groups))

    sub_start_idx = _convert_to_index(result, sub_start, npoints, default=0)
    if sub_start_idx is None: sub_start_idx = 0
    sub_end_idx = _convert_to_index(result, sub_end, npoints, default=npoints - 1)
    if sub_end_idx is None: sub_end_idx = npoints - 1
    
    fit_start_idx = _convert_to_index(result, fit_start, npoints, default=0)
    if fit_start_idx is None: fit_start_idx = 0
    fit_end_idx = _convert_to_index(result, fit_end, npoints, default=npoints - 1)
    if fit_end_idx is None: fit_end_idx = npoints - 1
    

    def poly_model(x, *coeffs):
        order = len(coeffs) - 1
        y = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(coeffs):
            y += c * x ** (order - i)
        return y

    def fit_and_subtract_freq(vector: np.ndarray) -> np.ndarray:
        if node_groups:
            x_fit = np.array(node_groups)
            y_fit = vector[node_groups]
        else:
            x_fit = np.arange(fit_start_idx, fit_end_idx + 1)
            y_fit = vector[fit_start_idx:fit_end_idx + 1]

        threshold = float(np.median(y_fit) + 1 * np.std(y_fit))
        mask = y_fit < threshold

        x_fit_masked = x_fit[mask]
        y_fit_masked = y_fit[mask]

        if len(x_fit_masked) < order + 1:
            return vector

        x_subtract = np.arange(sub_start_idx, sub_end_idx + 1).astype(np.float64)
        p0 = np.ones(order + 1)
        coeffs, _ = curve_fit(poly_model, x_fit_masked, y_fit_masked, p0=p0)
        baseline = poly_model(x_subtract, *coeffs)

        corrected = vector.copy()
        corrected[sub_start_idx:sub_end_idx + 1] -= baseline
        return corrected

    corrected_data = np.apply_along_axis(fit_and_subtract_freq, axis=-1, arr=data)

    if isinstance(corrected_data, NMRData):
        elapsed = perf_counter() - start_time
        corrected_data.processing_history.append({
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
        })

    return cast(NMRArrayType, corrected_data)


def polynomial_baseline_correction(
    data: NMRArrayType,
    *,
    sub_start: int = 0,
    sub_end: int = -1,
    fit_start: int = 0,
    fit_end: int = -1,
    start: int | None = None,
    end: int | None = None,
    node_list: list[str | int | None] | None,
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
    sx1: int | None = None,
    sxn: int | None = None,
    fx1: int | None = None,
    fxn: int | None = None,
    x1: int | None = None,
    xn: int | None = None,
    nl: list[str | int | None] | None,
    nw: int | None = None,
    ord: int | None = None,
    nc: int | None = None,
    first: bool | None = None,
    last: bool | None = None,
    avg: bool | None = None,
    filt: bool | None = None,
    
    time: bool | None = None,
    window: int | None = None,
    frac: float | None = None,
    nf: float | None = None,
    noise: float | None = None,
    #noseq: bool | None = None,
    #nodmx: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Switch domain
    if time is not None: domain = "time"
    
    if domain == "time":
        # Handle aliases
        noise_window_size = window if window is not None else noise_window_size
        min_baseline_fraction = frac if frac is not None else min_baseline_fraction
        noise_adjustment_factor = nf if nf is not None else noise_adjustment_factor
        rms_noise_value = noise if noise is not None else rms_noise_value

        
        return _pbc_time(
            data,
            order=order,
            noise_window_size=noise_window_size,
            min_baseline_fraction=min_baseline_fraction,
            noise_adjustment_factor=noise_adjustment_factor,
            rms_noise_value=rms_noise_value,
        )
    
    elif domain == "frequency":
        # Handle aliases
        sub_start = sx1 if sx1 is not None else sub_start
        sub_end = sxn if sxn is not None else sub_end
        fit_start = fx1 if fx1 is not None else fit_start
        fit_end = fxn if fxn is not None else fit_end
        start = x1 if x1 is not None else start
        end = xn if xn is not None else end
        node_list = nl if nl is not None else node_list
        node_width = nw if nw is not None else node_width
        order = ord if ord is not None else order
        initial_fit_nodes = nc if nc is not None else initial_fit_nodes
        use_first_points = first if first is not None else use_first_points
        use_last_points = last if last is not None else use_last_points
        use_node_avg = avg if avg is not None else use_node_avg
        sine_filter = filt if filt is not None else sine_filter



        
        # Overwrite subtraction region range and fit region range
        if start is not None:
            sub_start = start
            fit_start = start
        if end is not None:
            sub_end = end
            fit_end = end
            
        if sub_end == -1: sub_end = data.shape[-1] - 1
        if fit_end == -1: fit_end = data.shape[-1] - 1
        
        return _pbc_freq(
            data,
            sub_start=sub_start,
            sub_end=sub_end,
            fit_start=fit_start,
            fit_end=fit_end,
            node_list=node_list,
            node_width=node_width,
            order=order,
            use_first_points=use_first_points,
            use_last_points=use_last_points,
            use_node_avg=use_node_avg,
            sine_filter=sine_filter,
        )
        
    else:
        raise ValueError(f"Unknown domain '{domain}'. Must be 'time' or 'frequency'.")



# NMRPipe alias
POLY = polynomial_baseline_correction
POLY.__doc__ = polynomial_baseline_correction.__doc__  # Auto-generated
POLY.__name__ = "POLY"  # Auto-generated



def transpose(
    data: NMRArrayType,
    *,
    axes: list[int] | None = None,
    hyper_complex: bool = False,
    # Aliases
    hyper: bool = False,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    hyper_complex = hyper if hyper is not None else hyper_complex

    
    if hyper_complex:
        raise NotImplementedError("Hyper complex transpose is not yet implemented.")
    
    ndim = data.ndim
    if axes is None:
        axes = list(reversed(range(ndim)))
    
    result = data.transpose(*axes)

        

    if isinstance(data, NMRData):
        new_result = NMRData(result, copy_from=data)

        # Reorder axes metadata
        new_result.axes = [copy.deepcopy(data.axes[i]) for i in axes]

        # Update 'interleaved_data' handling if present
        if new_result.axes[-1].get("interleaved_data", False):
            new_result = _interleaved_to_complex(new_result)
            new_result.axes[-1]["interleaved_data"] = False

        # Append to processing history
        elapsed = perf_counter() - start_time
        new_result.processing_history.append({
            'Function': "Transpose",
            'axes': list(axes),
            'shape_before': data.shape,
            'shape_after': new_result.shape,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })

        return cast(NMRArrayType, new_result)

    return cast(NMRArrayType, result)

# NMRPipe alias
TP = transpose
TP.__doc__ = transpose.__doc__  # Auto-generated
TP.__name__ = "TP"  # Auto-generated

ZTP = transpose
ZTP.__doc__ = transpose.__doc__  # Auto-generated
ZTP.__name__ = "ZTP"  # Auto-generated


def add_constant(
    data: NMRArrayType,
    *,
    start: str | int | None = None,
    end: str | int | None = None,
    constant: float | None = None,
    constant_real: float | None = None,
    constant_imaginary: float | None = None,
    # Alias
    r: float | None = None,
    i: float | None = None,
    c: float | None = None,
    x1: str | int | None = None,
    xn: str | int | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    constant_real = r if r is not None else constant_real
    constant_imaginary = i if i is not None else constant_imaginary
    constant = c if c is not None else constant
    start = x1 if x1 is not None else start
    end = xn if xn is not None else end

    
    
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


    if isinstance(data, NMRData):
        result = NMRData(array, copy_from=data)

        elapsed = perf_counter() - start_time
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
        
        return cast(NMRArrayType, result)

    return cast(NMRArrayType, array)

# NMRPipe alias
ADD = add_constant
ADD.__doc__ = add_constant.__doc__  # Auto-generated
ADD.__name__ = "ADD"  # Auto-generated


def multiply_constant(
    data: NMRArrayType,
    *,
    start: str | int | None = None,
    end: str | int | None = None,
    constant: float | None = None,
    constant_real: float | None = None,
    constant_imaginary: float | None = None,
    # Alias
    r: float | None = None,
    i: float | None = None,
    c: float | None = None,
    x1: str | int | None = None,
    xn: str | int | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    constant_real = r if r is not None else constant_real
    constant_imaginary = i if i is not None else constant_imaginary
    constant = c if c is not None else constant
    start = x1 if x1 is not None else start
    end = xn if xn is not None else end



    
    
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

    if isinstance(data, NMRData):
        result = NMRData(array, copy_from=data)

        elapsed = perf_counter() - start_time
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

    return cast(NMRArrayType, array)

# NMRPipe alias
MULT = multiply_constant
MULT.__doc__ = multiply_constant.__doc__  # Auto-generated
MULT.__name__ = "MULT"  # Auto-generated


def set_to_constant(
    data: NMRArrayType,
    *,
    start: str | int | None = None,
    end: str | int | None = None,
    constant: float = 0.0,
    constant_real: float = 0.0,
    constant_imaginary: float = 0.0,
    # Alias
    r: float | None = None,
    i: float | None = None,
    c: float | None = None,
    x1: str | int | None = None,
    xn: str | int | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    constant_real = r if r is not None else constant_real
    constant_imaginary = i if i is not None else constant_imaginary
    constant = c if c is not None else constant
    start = x1 if x1 is not None else start
    end = xn if xn is not None else end



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


    if isinstance(data, NMRData):
        result = NMRData(array, copy_from=data)

        elapsed = perf_counter() - start_time
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
        
        return cast(NMRArrayType, result)

    return cast(NMRArrayType, array)

# NMRPipe alias
SET = set_to_constant
SET.__doc__ = set_to_constant.__doc__  # Auto-generated
SET.__name__ = "SET"  # Auto-generated


def delete_imaginaries(data: NMRArrayType) -> NMRArrayType:
    """
    Discard the imaginary part of complex-valued NMRData.

    Args:
        data (NMRData): Complex NMRData.

    Returns:
        NMRData: Real-valued data.
    """
    start_time = perf_counter()

    # Take the real part only
    real_data = np.real(data).copy()

    # Create new NMRData object with real data and preserved metadata
    if isinstance(data, NMRData):
        result = NMRData(real_data, copy_from=data)

        # Record processing history
        elapsed = perf_counter() - start_time
        result.processing_history.append({
            'Function': "Delete imaginary part",
            'imag_removed': True,
            'dtype_before': str(data.dtype),
            'dtype_after': str(real_data.dtype),
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })
        return cast(NMRArrayType, result)

    return cast(NMRArrayType, real_data)

# NMRPipe alias
DI = delete_imaginaries
DI.__doc__ = delete_imaginaries.__doc__  # Auto-generated
DI.__name__ = "DI"  # Auto-generated


def null(data: NMRArrayType) -> NMRArrayType:
    """
    Leave data unchanged.

    Args:
        data (NMRData): Input data.

    Returns:
        NMRData: Unchagned NMRData.
    """
    start_time = perf_counter()
    
    result = data.copy()
    
    if isinstance(result, NMRData):
        elapsed = perf_counter() - start_time
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
    data: NMRArrayType,
    *,
    adjust_spectral_width: bool = True,
    # Aliases
    sw: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    adjust_spectral_width = sw if sw is not None else adjust_spectral_width

    
    
    if adjust_spectral_width:
        result = data[..., ::-1]
    
    else:
        reversed_array = data[..., ::-1]
        
        if isinstance(data, NMRData):
            result = NMRData(reversed_array, copy_from=data)
        else:
            result = reversed_array


    if isinstance(result, NMRData):
        elapsed = perf_counter() - start_time
        result.processing_history.append({
            'Function': "Reverse data",
            'adjusted_sw': adjust_spectral_width,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })

    return cast(NMRArrayType, result)

# NMRPipe alias
REV = reverse
REV.__doc__ = reverse.__doc__  # Auto-generated
REV.__name__ = "REV"  # Auto-generated


def right_shift(
    data: NMRArrayType,
    *,
    shift_amount: str | int = 0,
    adjust_spectral_width: bool = True,
    # Aliases
    rs:  str | int | None = None,
    sw: bool | None = None,
) -> NMRArrayType:
    """
    Apply a right shift and zero pad to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        shift_amount (int | str): Amount to shift (e.g. 10, "3 ppm", "1000 pts", "10%").
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after shifting.
        
    Aliases:
        rs: Alias for shift_amount.
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Data after applying right shift and zero padding.
    """
    start_time = perf_counter()
    
    # Handle aliases
    shift_amount = rs if rs is not None else shift_amount
    adjust_spectral_width = sw if sw is not None else adjust_spectral_width

    
    dim = -1 
    npoints = data.shape[dim]
    
    shift_points = _convert_to_index(data, shift_amount, npoints, default=0)
    
    shift_points = int(np.clip(shift_points, -npoints, npoints))
    
    shifted_data = np.zeros_like(data)
    if shift_points > 0:
        shifted_data[..., shift_points:] = data[..., :-shift_points]
    
    elif shift_points < 0:
        shifted_data[..., :shift_points] = data[..., -shift_points:]
    
    else:
        shifted_data = data.copy()
    
    if isinstance(data, NMRData):
        result = NMRData(shifted_data, copy_from=data)

        if adjust_spectral_width:
            sw, ori, obs = (data.axes[dim][k] for k in ("SW", "ORI", "OBS"))
            if sw is None or ori is None or obs is None:
                raise ValueError(f"Missing SW, ORI, or OBS in axis {dim} metadata. Found: SW={sw}, ORI={ori}, OBS={obs}")
            
            unit = data.axes[dim].get("unit", "pts").lower()
            point_shift = sw / npoints

            if unit == "pts":
                new_ori = ori - (point_shift * shift_points)
            else:
                ppm_scale = get_ppm_scale(npoints, sw, ori, obs)
                new_ppm_scale = np.roll(ppm_scale, shift_points)
                new_ori = obs * new_ppm_scale[0]

            axis = data.axes[dim].copy()
            axis["ORI"] = new_ori
            axis["scale"] = np.roll(axis["scale"], shift_points)
            result.axes[dim] = axis

        result.processing_history.append({
            'Function': "Right shift data",
            'shift_amount': shift_amount,
            'adjusted_sw': adjust_spectral_width,
            'time_elapsed_s': perf_counter() - start_time,
            'time_elapsed_str': _format_elapsed_time(perf_counter() - start_time),
        })
    else:
        result = shifted_data

    return cast(NMRArrayType, result)

# NMRPipe alias
RS = right_shift
RS.__doc__ = right_shift.__doc__  # Auto-generated
RS.__name__ = "RS"  # Auto-generated


def left_shift(
    data: NMRArrayType,
    *,
    shift_amount: str | int = 0,
    adjust_spectral_width: bool = True,
    # Aliases
    ls:  str | int | None = None,
    sw: bool | None = None,
) -> NMRArrayType:
    """
    Apply a left shift and zero pad to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        shift_amount (int | str): Amount to shift (e.g. 10, "3 ppm", "1000 pts", "10%").
        adjust_spectral_width (bool): If True, adjust SW, ORI, and OBS metadata after shifting.
        
    Aliases:
        ls: Alias for shift_amount.
        sw: Alias for adjust_spectral_width.

    Returns:
        NMRData: Data after applying right shift and zero padding.
    """
    start_time = perf_counter()
    
    # Handle aliases
    shift_amount = ls if ls is not None else shift_amount
    adjust_spectral_width = sw if sw is not None else adjust_spectral_width

    
    shift_points = _convert_to_index(data, shift_amount, data.shape[-1], default=0)
    
    result = right_shift(data, shift_amount=-shift_points, adjust_spectral_width=adjust_spectral_width)
    
    if isinstance(result, NMRData):
        result.processing_history.pop()

        elapsed = perf_counter() - start_time
        result.processing_history.append(
            {
                'Function': "Left shift data",
                'shift_amount': shift_amount,
                'adjusted_sw': adjust_spectral_width,
                'time_elapsed_s': elapsed,
                'time_elapsed_str': _format_elapsed_time(elapsed),
            }
        )

    return cast(NMRArrayType, result)

# NMRPipe alias
LS = left_shift
LS.__doc__ = left_shift.__doc__  # Auto-generated
LS.__name__ = "LS"  # Auto-generated


def circular_shift(
    data: NMRArrayType,
    *,
    right_shift_amount: str | int = 0,
    left_shift_amount:  str | int = 0,
    negate_shifted: bool = False,
    adjust_spectral_width: bool = True,
    # Aliases
    rs:  str | int | None = None,
    ls:  str | int | None = None,
    neg: bool | None = None,
    sw: bool | None = None,
) -> NMRArrayType:
    """
    Apply a circular shift to the data in the last dimension.

    Args:
        data (NMRData): Input NMR dataset.
        right_shift_amount (int | str): Shift right by this amount (e.g., 100, "5 ppm").
        left_shift_amount (int | str): Shift left by this amount.
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
    start_time = perf_counter()
    
    # Handle aliases
    right_shift_amount = rs if rs is not None else right_shift_amount
    left_shift_amount = ls if ls is not None else left_shift_amount
    negate_shifted = neg if neg is not None else negate_shifted
    adjust_spectral_width = sw if sw is not None else adjust_spectral_width

    
    if right_shift_amount != 0 and left_shift_amount != 0:
        raise ValueError("Specify only one of right_shift_amount (rs) or left_shift_amount (ls), not both.")
    
    dim = -1
    npoints = data.shape[dim]
    
    shift_spec = right_shift_amount if right_shift_amount else f"-{left_shift_amount}"
    shift_points = _convert_to_index(data, shift_spec, npoints, default=0)
    shift_points = shift_points % npoints
    
    
    shifted_array = np.roll(data, shift_points, axis=dim)
    
    if negate_shifted and shift_points > 0:
        slicer = [slice(None)] * data.ndim
        slicer[dim] = slice(0, shift_points)
        shifted_array[tuple(slicer)] *= -1
    
    if isinstance(data, NMRData):
        new_data = NMRData(shifted_array, copy_from=data)

        if adjust_spectral_width:
            axis = data.axes[dim]
            sw_value = axis.get("SW")
            ori = axis.get("ORI")
            obs = axis.get("OBS")

            if sw_value is None or ori is None or obs is None:
                raise ValueError(f"Missing SW, ORI, or OBS in axis {dim} metadata.")

            point_shift = sw_value / npoints
            new_ori = ori - point_shift * shift_points

            axis_new = axis.copy()
            axis_new["ORI"] = new_ori
            axis_new["scale"] = np.roll(axis["scale"], shift_points)
            axis_new["SW"] = sw_value
            axis_new["OBS"] = obs

            new_data.axes[dim] = axis_new

        new_data.processing_history.append({
            'Function': "Circular shift data",
            'shift_amount': shift_points,
            'negate_shifted': negate_shifted,
            'adjusted_sw': adjust_spectral_width,
            'time_elapsed_s': perf_counter() - start_time,
            'time_elapsed_str': _format_elapsed_time(perf_counter() - start_time),
        })

    else:
        new_data = shifted_array

    return cast(NMRArrayType, new_data)

# NMRPipe alias
CS = circular_shift
CS.__doc__ = circular_shift.__doc__  # Auto-generated
CS.__name__ = "CS"  # Auto-generated


def manipulate_sign(
    data: NMRArrayType,
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
    ri: bool | None = None,
    r: bool | None = None,
    i: bool | None = None,
    left: bool | None = None,
    right: bool | None = None,
    alt: bool | None = None,
    abs: bool | None = None,
    sign: bool | None = None,
) -> NMRArrayType:
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
    start_time = perf_counter()
    
    # Handle aliases
    negate_all = ri if ri is not None else negate_all
    negate_reals = r if r is not None else negate_reals
    negate_imaginaries = i if i is not None else negate_imaginaries
    negate_left_half = left if left is not None else negate_left_half
    negate_right_half = right if right is not None else negate_right_half
    alternate_sign = alt if alt is not None else alternate_sign
    absolute_value = abs if abs is not None else absolute_value
    replace_with_sign = sign if sign is not None else replace_with_sign



    
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

    if isinstance(result, NMRData):
        elapsed = perf_counter() - start_time
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

    return cast(NMRArrayType, result)

# NMRPipe alias
SIGN = manipulate_sign
SIGN.__doc__ = manipulate_sign.__doc__  # Auto-generated
SIGN.__name__ = "SIGN"  # Auto-generated


def modulus(
    data: NMRArrayType,
    *,
    modulus: bool = True,
    modulus_squared: bool = False,
    # Aliases
    mod: bool | None = None,
    pow: bool | None = None,
) -> NMRArrayType:
    """
    Compute the modulus (magnitude) or squared modulus of complex NMR data.

    Args:
        data (NMRData or np.ndarray): Complex-valued input NMR dataset.
        modulus (bool): If True, compute the modulus (|z|).
        modulus_squared (bool): If True, compute the squared modulus (|z|^2).

    Aliases:
        mod: Alias for modulus.
        pow: Alias for modulus_squared.

    Returns:
        NMRData or np.ndarray: Data after applying modulus or modulus squared.
    """
    start_time = perf_counter()
    
    # Handle aliases
    modulus = mod if mod is not None else modulus
    modulus_squared = pow if pow is not None else modulus_squared

    
    if modulus_squared:
        result_array = np.real(data) ** 2 + np.imag(data) ** 2
    else:
        result_array = np.abs(data)


    if isinstance(data, NMRData):
        result = NMRData(result_array, copy_from=data)
        elapsed = perf_counter() - start_time
        result.processing_history.append({
            'Function': "Modulus",
            'modulus': modulus,
            'modulus_squared': modulus_squared,
            'time_elapsed_s': elapsed,
            'time_elapsed_str': _format_elapsed_time(elapsed),
        })
        return result

    return cast(NMRArrayType, result_array)

# NMRPipe alias
MC = modulus
MC.__doc__ = modulus.__doc__  # Auto-generated
MC.__name__ = "MC"  # Auto-generated