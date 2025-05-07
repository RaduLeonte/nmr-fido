from __future__ import annotations
from typing import Callable
import numpy as np
import copy
import inspect
import functools

from nmr_fido.utils import get_ppm_scale


class NMRData(np.ndarray):
    # Declare so IDE can autocomplete
    axes: list[dict]
    metadata: dict
    processing_history: list[dict]
    
    _custom_attrs = ['axes', 'metadata', 'processing_history']
    
    def __new__(
        cls,
        input_array: np.ndarray,
        axes: list[dict] = None,
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
        
        # Calculate ppm scale
        ppm_scale = get_ppm_scale(npoints, sw, ori, obs)
        

        self.scales[dim] = ppm_scale
        self.units[dim] = "ppm"

        return self