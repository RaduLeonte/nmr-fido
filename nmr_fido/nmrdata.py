from __future__ import annotations
from typing import Callable
import numpy as np
import copy
import inspect
import functools


class NMRData(np.ndarray):
    # Declare so IDE can autocomplete
    labels: list[str]
    scales: list[np.ndarray]
    units: list[str]
    axis_info: list[dict]
    metadata: dict
    processing_history: list[dict]
    
    _custom_attrs = ['labels', 'scales', 'units', 'axis_info', 'metadata', 'processing_history']
    
    def __new__(
        cls,
        input_array: np.ndarray,
        labels: list[str] = None,
        scales: list[np.ndarray] = None,
        units: list[str] = None,
        axis_info: list[dict] = None,
        metadata: dict = None,
        processing_history: list[dict] = None,
        copy_from: NMRData = None,
    ):
        """
        Create a new NMRData object, a subclass of numpy.ndarray with NMR specific metadata.

        Parameters:
            input_array (np.ndarray):
                The raw data array. This must be a numpy compatible array (e.g., from np.zeros or np.ones).

            labels (list of str, optional):
                Human-readable names for each axis (e.g., ['15N', '13C', '1H']).

            scales (list of np.ndarray, optional):
                Coordinate scales for each axis (e.g., ppm or time values). Each entry must match the size of the corresponding dimension.

            units (list of str, optional):
                Unit strings for each axis (e.g., 'ppm', 'Hz', 'pts').

            axis_info (list of dict, optional):
                Metadata for each axis (e.g., {'SW': 8000, 'SF': 600, 'OFFSET': 4.7}).

            metadata (dict, optional):
                Global metadata for the dataset, such as acquisition parameters.

            processing_history (list of dict, optional):
                A list of dictionaries describing each processing step applied to the data.

            copy_from (NMRData, optional):
                An existing NMRData object to inherit all metadata from (except the data array).

        Returns:
            NMRData:
                A NumPy ndarray with enhanced metadata support for NMR processing and visualization.

        NOTE:
            The axis ordering follows the convention: the **last axis is the fastest-varying (X), and the first is the slowest (Z)**.
            That means: **dimension order is Z (outermost), Y (middle), X (innermost)**.
            So, when specifying `labels`, `scales`, or `units`, ensure that:
                labels = ['Z-axis', 'Y-axis', 'X-axis'] — NOT ['X', 'Y', 'Z']
        """
        obj: np.ndarray = np.asarray(input_array).view(cls)
        
        
        init_args = {
            'labels': labels,
            'scales': scales,
            'units': units,
            'axis_info': axis_info,
            'metadata': metadata,
            'processing_history': processing_history
        }
        
        for attr in cls._custom_attrs:
            if init_args.get(attr) is not None:
                setattr(obj, attr, copy.deepcopy(init_args[attr]))
            elif copy_from is not None and getattr(copy_from, attr, None) is not None:
                setattr(obj, attr, copy.deepcopy(getattr(copy_from, attr)))
            else:
                setattr(obj, attr, cls._default_value(attr, input_array))

        return obj
    
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in self._custom_attrs:
            setattr(self, attr, getattr(obj, attr, None))
    
    
    @staticmethod
    def _default_value(attr: str, input_array: np.ndarray):
        if attr == 'labels':
            return [f"Axis {i}" for i in range(input_array.ndim)]
        elif attr == 'scales':
            return [np.arange(size) for size in input_array.shape]
        elif attr == 'units':
            return ["pts"] * input_array.ndim
        elif attr == 'axis_info':
            return [{} for _ in range(input_array.ndim)]
        elif attr == 'metadata':
            return {}
        elif attr == 'processing_history':
            return []
        else:
            return None
    
    
    def __getitem__(self, item) -> NMRData:
        result = super().__getitem__(item)

        if not isinstance(result, np.ndarray):
            return result

        # Wrap result as NMRData
        result = result.view(type(self))

        # Expand item to full list of slicers
        if isinstance(item, tuple):
            slicers = list(item) + [slice(None)] * (self.ndim - len(item))
        else:
            slicers = [item] + [slice(None)] * (self.ndim - 1)

        surviving_dims = [i for i, s in enumerate(slicers) if not isinstance(s, int)]

        for attr in self._custom_attrs:
            value = getattr(self, attr, None)

            match attr:
                case 'scales':
                    if value is not None and isinstance(value, list):
                        new_scales = []
                        for i, s in enumerate(slicers):
                            if i >= len(value):
                                continue  # prevent IndexError
                            if isinstance(s, slice):
                                new_scales.append(value[i][s])
                            elif isinstance(s, int):
                                pass  # axis removed
                            else:
                                new_scales.append(value[i])
                        setattr(result, attr, new_scales)
                    else:
                        setattr(result, attr, copy.deepcopy(value))

                case 'labels' | 'units' | 'axis_info':
                    if isinstance(value, list):
                        # Avoid IndexError by checking list length
                        setattr(result, attr, [value[i] for i in surviving_dims if i < len(value)])
                    else:
                        setattr(result, attr, copy.deepcopy(value))

                case _:
                    setattr(result, attr, copy.deepcopy(value))

        return result


    def __str__(self) -> str:
        lines = [
            f"<NMRData shape={self.shape} dtype={self.dtype}>",
            "Axes:"
        ]

        if (
            hasattr(self, 'labels') and
            hasattr(self, 'scales') and
            hasattr(self, 'units')
        ):
            for i, (label, scale, unit) in enumerate(zip(self.labels, self.scales, self.units)):
                range_str = f"{scale[0]:.2f} to {scale[-1]:.2f}" if scale.size > 0 else "empty"
                lines.append(f" {len(self.data.shape) - i} {label}: Size={scale.size}, Range=({range_str}), Unit={unit}")

        metadata_keys = list(self.metadata.keys()) if hasattr(self, 'metadata') and self.metadata else []
        lines.append(f"Metadata keys: {metadata_keys[:5]}")

        return "\n".join(lines)
    
    __repr__ = __str__
    
    
    def _update_from(self, other: NMRData):
        """Helper to update self's contents from another NMRData object."""
        self.resize(other.shape, refcheck=False)
        np.copyto(self, other)
        for attr in self._custom_attrs:
            setattr(self, attr, copy.deepcopy(getattr(other, attr)))
            
    
    @classmethod
    def _add_processing_method(cls, func: Callable, method_name: str = None):
        """
        Dynamically add a processing method to NMRData.

        Args:
            func (Callable): The processing function (e.g., zero_fill)
            method_name (str): Optional method name (default = func.__name__)
        """
        name = method_name or func.__name__

        # Create a wrapper that preserves docstring and name
        @functools.wraps(func)
        def method(self, **kwargs):
            return func(self, **kwargs)

        method.__name__ = name
        method.__doc__ = func.__doc__

        # Inject original function signature for IDEs
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if params and params[0].name == "data":
                params[0] = params[0].replace(name="self")
            method.__signature__ = sig.replace(parameters=params)
        
        except Exception:
            pass  # fallback if signature injection fails
        
        setattr(cls, name, method)
    
    
    def scale_to_ppm(self, target_dim: int = -1) -> NMRData:
        """
        Convert the scale of the target dimension to ppm using axis_info.

        Args:
            target_dim (int): Which dimension to convert. Defaults to last dimension (-1).

        Returns:
            NMRData: The updated object (self).
        """
        dim = target_dim if target_dim >= 0 else self.ndim + target_dim

        if not hasattr(self, 'axis_info') or len(self.axis_info) <= dim:
            raise ValueError(f"No axis_info available for dimension {dim}.")

        info = self.axis_info[dim]
        
        if not all(k in info for k in ("SW", "ORI", "OBS")):
            return self

        sw = info["SW"] # Sweep width [Hz]
        ori = info["ORI"] # Origin freq (middle of spectrum) [Hz]
        obs = info["OBS"] # Observer frequency (spectrometer freq.) [MHz]

        npoints = self.shape[dim]
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
        

        self.scales[dim] = ppm_scale
        self.units[dim] = "ppm"

        return self
    
    # region Processing stubs
    # Auto-generated processing stubs for IDE support
    def solvent_filter(self, ) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): Input data.
        Aliases:
        Returns:
            NMRData: .
        """
        ...
    def SOL(self, ) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): Input data.
        Aliases:
        Returns:
            NMRData: .
        """
        ...
    def linear_prediction(self, *, pred: int = None, x1: int = None, xn: int = None, ord: int = None, f: bool = None, b: bool = None, fb: bool = None, before: bool = None, after: bool = None, nofix: bool = None, fix: bool = None, fixMode: int = None, ps90_180: bool = None, ps0_0: bool = None, pca: bool = None, extra: int = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): Input data.
        Aliases:
        Returns:
            NMRData: .
        """
        ...
    def LP(self, *, pred: int = None, x1: int = None, xn: int = None, ord: int = None, f: bool = None, b: bool = None, fb: bool = None, before: bool = None, after: bool = None, nofix: bool = None, fix: bool = None, fixMode: int = None, ps90_180: bool = None, ps0_0: bool = None, pca: bool = None, extra: int = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): Input data.
        Aliases:
        Returns:
            NMRData: .
        """
        ...
    def sine_bell_window(self, *, start_angle: float = 0.0, end_angle: float = 1.0, exponent: float = 1.0, size_window: int = None, start: int = 1, scale_factor_first_point: float = 1.0, fill_outside_one: bool = False, invert_window: bool = False, off: float = None, end: float = None, pow: float = None, size: int = None, c: float = None, one: bool = None, inv: bool = None) -> NMRData:
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
        ...
    def SP(self, *, start_angle: float = 0.0, end_angle: float = 1.0, exponent: float = 1.0, size_window: int = None, start: int = 1, scale_factor_first_point: float = 1.0, fill_outside_one: bool = False, invert_window: bool = False, off: float = None, end: float = None, pow: float = None, size: int = None, c: float = None, one: bool = None, inv: bool = None) -> NMRData:
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
        ...
    def zero_fill(self, *, factor: int = 1, add: int = None, final_size: int = None, zf: int = None, pad: int = None, size: int = None) -> NMRData:
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
        ...
    def ZF(self, *, factor: int = 1, add: int = None, final_size: int = None, zf: int = None, pad: int = None, size: int = None) -> NMRData:
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
        ...
    def fourier_transform(self, *, real_only: bool = False, inverse: bool = False, negate_imaginaries: bool = False, sign_alteration: bool = False, bruk: bool = False, norm: str = 'backward', real: bool = None, inv: bool = None, neg: bool = None, alt: bool = None) -> NMRData:
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
        ...
    def FT(self, *, real_only: bool = False, inverse: bool = False, negate_imaginaries: bool = False, sign_alteration: bool = False, bruk: bool = False, norm: str = 'backward', real: bool = None, inv: bool = None, neg: bool = None, alt: bool = None) -> NMRData:
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
        ...
    def hilbert_transform(self, *, mirror_image: bool = False, temporary_zero_fill: bool = False, size_time_domain: int = None, ps90_180: bool = None, zf: bool = None, td: bool = None) -> NMRData:
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
        ...
    def HT(self, *, mirror_image: bool = False, temporary_zero_fill: bool = False, size_time_domain: int = None, ps90_180: bool = None, zf: bool = None, td: bool = None) -> NMRData:
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
        ...
    def phase(self, *, p0: float = 0.0, p1: float = 0.0, invert: bool = False, reconstruct_imaginaries: bool = False, temporary_zero_fill: bool = False, exponential_correction: bool = False, decay_constant: float = 0.0, inv: float = None, ht: bool = None, zf: bool = None, exp: bool = None, tc: float = None) -> NMRData:
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
        ...
    def PS(self, *, p0: float = 0.0, p1: float = 0.0, invert: bool = False, reconstruct_imaginaries: bool = False, temporary_zero_fill: bool = False, exponential_correction: bool = False, decay_constant: float = 0.0, inv: float = None, ht: bool = None, zf: bool = None, exp: bool = None, tc: float = None) -> NMRData:
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
        ...
    def extract_region(self, *, start: str | int = None, end: str | int = None, start_y: str | int = None, end_y: str | int = None, left_half: bool = False, right_half: bool = False, middle_half: bool = False, power_of_two: bool = False, multiple_of: int = None, left: bool = None, right: bool = None, mid: bool = None, pow2: bool = None, round: int = None, x1: str = None, xn: str = None, y1: str = None, yn: str = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): The data to extract region from.
        Returns:
            NMRData: Desc.
        """
        ...
    def EXT(self, *, start: str | int = None, end: str | int = None, start_y: str | int = None, end_y: str | int = None, left_half: bool = False, right_half: bool = False, middle_half: bool = False, power_of_two: bool = False, multiple_of: int = None, left: bool = None, right: bool = None, mid: bool = None, pow2: bool = None, round: int = None, x1: str = None, xn: str = None, y1: str = None, yn: str = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): The data to extract region from.
        Returns:
            NMRData: Desc.
        """
        ...
    def polynomial_baseline_correction(self, *, domain: str = 'freq', order: int = 4, sx1: int = None, sxn: int = None, fx1: int = None, fxn: int = None, x1: int = None, xn: int = None, nl: int = None, nw: int = None, ord: int = None, nc: int = None, first: int = None, last: int = None, avg: int = None, filt: int = None, time: int = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): The data to transpose.
        Returns:
            NMRData: .
        """
        ...
    def POLY(self, *, domain: str = 'freq', order: int = 4, sx1: int = None, sxn: int = None, fx1: int = None, fxn: int = None, x1: int = None, xn: int = None, nl: int = None, nw: int = None, ord: int = None, nc: int = None, first: int = None, last: int = None, avg: int = None, filt: int = None, time: int = None) -> NMRData:
        """
        Desc.
        Args:
            data (NMRData): The data to transpose.
        Returns:
            NMRData: .
        """
        ...
    def transpose(self, *, axes: list[int] = None) -> NMRData:
        """
        Transpose the data and reorder metadata accordingly.
        Args:
            data (NMRData): The data to transpose.
            axes (list[int], optional): New axis order. If None, reverse axes.
        Returns:
            NMRData: Transposed data.
        """
        ...
    def TP(self, *, axes: list[int] = None) -> NMRData:
        """
        Transpose the data and reorder metadata accordingly.
        Args:
            data (NMRData): The data to transpose.
            axes (list[int], optional): New axis order. If None, reverse axes.
        Returns:
            NMRData: Transposed data.
        """
        ...
    def ZTP(self, *, axes: list[int] = None) -> NMRData:
        """
        Transpose the data and reorder metadata accordingly.
        Args:
            data (NMRData): The data to transpose.
            axes (list[int], optional): New axis order. If None, reverse axes.
        Returns:
            NMRData: Transposed data.
        """
        ...
    def add_constant(self, *, start: str | int = None, end: str | int = None, constant: float = None, constant_real: float = None, constant_imaginary: float = None, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def ADD(self, *, start: str | int = None, end: str | int = None, constant: float = None, constant_real: float = None, constant_imaginary: float = None, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def multiply_constant(self, *, start: str | int = None, end: str | int = None, constant: float = None, constant_real: float = None, constant_imaginary: float = None, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def MULT(self, *, start: str | int = None, end: str | int = None, constant: float = None, constant_real: float = None, constant_imaginary: float = None, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def set_to_constant(self, *, start: str | int = None, end: str | int = None, constant: float = 0.0, constant_real: float = 0.0, constant_imaginary: float = 0.0, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def SET(self, *, start: str | int = None, end: str | int = None, constant: float = 0.0, constant_real: float = 0.0, constant_imaginary: float = 0.0, r: float = None, i: float = None, c: float = None, x1: str | int = None, xn: str | int = None) -> NMRData:
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
        ...
    def delete_imaginaries(self, ) -> NMRData:
        """
        Discard the imaginary part of complex-valued NMRData.
        Args:
            data (NMRData): Complex NMRData.
        Returns:
            NMRData: Real-valued data.
        """
        ...
    def DI(self, ) -> NMRData:
        """
        Discard the imaginary part of complex-valued NMRData.
        Args:
            data (NMRData): Complex NMRData.
        Returns:
            NMRData: Real-valued data.
        """
        ...
    # endregion Processing stubs




from nmr_fido.core import processing

for name in dir(processing):
    attr = getattr(processing, name)
    if callable(attr) and not name.startswith("_"):
        NMRData._add_processing_method(attr, method_name=name)