from __future__ import annotations
from typing import Callable
import numpy as np
import copy
import inspect
import functools

from nmr_fido.utils import get_hz_scale, get_ppm_scale


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

            axes (list of dict, optional):
                A list of dictionaries for each axis. Each dictionary can contain optional keys such as 'label', 'scale', 'units', 'SW', 'ORI', 'OBS'.
                If a key is not provided, a default value will be generated. For instance, 'scale' will default to a range array of the appropriate size.

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
            Each axis dictionary can include optional fields such as 'label', 'scale', 'units', 'SW', 'ORI', 'OBS'. Missing values will be populated with default values.
        """
        obj: np.ndarray = np.asarray(input_array).view(cls)
        
        init_args = {
            'axes': axes,
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

        # Populate missing axis entries with defaults
        if hasattr(obj, 'axes') and isinstance(obj.axes, list):
            for i, axis in enumerate(obj.axes):
                default_axis = cls._default_axis(i, input_array.shape[i])
                for key, value in default_axis.items():
                    axis.setdefault(key, value)

        return obj

    
    
    def __array_finalize__(self, obj):
        if obj is None: return
        
        for attr in self._custom_attrs:
            setattr(self, attr, getattr(obj, attr, None))
    
    
    @staticmethod
    def _default_value(attr: str, input_array: np.ndarray):
        if attr == 'axes':
            return [NMRData._default_axis(i, size) for i, size in enumerate(input_array.shape)]
        elif attr == 'metadata':
            return {}
        elif attr == 'processing_history':
            return []
        return None
    
    
    @staticmethod
    def _default_axis(index: int, size: int) -> dict:
        return {
            "label": f"Axis {index}",
            "scale": np.arange(size),
            "unit": "pts"
        }
    
    
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

        # Update axes attribute
        new_axes = []
        for i, s in enumerate(slicers):
            if i >= len(self.axes):
                continue  # prevent IndexError
            
            axis_dict = self.axes[i]
            
            if isinstance(s, slice):
                # Slice the scale data
                new_scale = axis_dict['scale'][s]
                new_axes.append({
                    'label': axis_dict['label'],
                    'scale': new_scale,
                    'unit': axis_dict['unit']
                })
            
            elif isinstance(s, int):
                # Drop axis
                pass
            
            else:
                new_axes.append(axis_dict)

        # Reorder axes based on surviving dimensions
        result.axes = [new_axes[i] for i in surviving_dims if i < len(new_axes)]

        # Copy other custom attributes
        for attr in self._custom_attrs:
            if attr != 'axes':
                setattr(result, attr, copy.deepcopy(getattr(self, attr, None)))

        return result


    def __str__(self) -> str:
        lines = [
            f'<NMRData shape={self.shape} dtype={self.dtype}">',
            "Data Preview:",
            str(np.array(self)),
        ]

        # Display axes information
        if hasattr(self, 'axes') and isinstance(self.axes, list):
            lines.append("Axes:")
            for i, axis in enumerate(self.axes):
                label = axis.get('label', f"Axis {i}")
                scale = axis.get('scale', [])
                size = len(scale)
                unit = axis.get('unit', '')
                range_str = f"{scale[0]:.2f}, {scale[-1]:.2f}" if len(scale) > 0 else "empty"
                lines.append(f" {len(self.shape) - i} {label}: Size={size}, Range=[{range_str}], Unit={unit}")

        # Display metadata keys
        #metadata_keys = list(self.metadata.keys()) if hasattr(self, 'metadata') and self.metadata else []
        #lines.append(f"Metadata keys: {metadata_keys[:5]}")

        return "\n".join(lines)
    
    __repr__ = __str__
    
    
    def _update_from(self, other: NMRData):
        """Helper to update self's contents from another NMRData object."""
        self.resize(other.shape, refcheck=False)
        np.copyto(self, other)
        for attr in self._custom_attrs:
            setattr(self, attr, copy.deepcopy(getattr(other, attr)))
    
    
    def scale_to_hz(self, target_dim: int = -1) -> NMRData:
        """
        Convert the scale of the target dimension to Hz using axis_info.

        Args:
            target_dim (int): Which dimension to convert. Defaults to last dimension (-1).

        Returns:
            NMRData: The updated object (self).
        """
        dim = target_dim if target_dim >= 0 else self.ndim + target_dim

        if not hasattr(self, 'axes') or len(self.axes) <= dim:
            raise ValueError(f"No axis_info available for dimension {dim}.")

        axis_dict = self.axes[dim]
        
        if not all(k in axis_dict for k in ("SW", "ORI")):
            return self

        sw = axis_dict["SW"] # Sweep width [Hz]
        ori = axis_dict["ORI"] # Origin freq (middle of spectrum) [Hz]

        npoints = self.shape[dim]
        
        hz_scale = get_hz_scale(npoints, sw, ori)

        self.axes[dim]["scale"] = hz_scale
        self.axes[dim]["unit"] = "Hz"

        return self
    
    
    
    def scale_to_ppm(self, target_dim: int = -1) -> NMRData:
        """
        Convert the scale of the target dimension to ppm using axis_info.

        Args:
            target_dim (int): Which dimension to convert. Defaults to last dimension (-1).

        Returns:
            NMRData: The updated object (self).
        """
        dim = target_dim if target_dim >= 0 else self.ndim + target_dim

        if not hasattr(self, 'axes') or len(self.axes) <= dim:
            raise ValueError(f"No axis_info available for dimension {dim}.")

        axis_dict = self.axes[dim]
        
        if not all(k in axis_dict for k in ("SW", "ORI", "OBS")):
            return self

        sw = axis_dict["SW"] # Sweep width [Hz]
        ori = axis_dict["ORI"] # Origin freq (middle of spectrum) [Hz]
        obs = axis_dict["OBS"] # Observer frequency (spectrometer freq.) [MHz]

        npoints = self.shape[dim]
        
        # Calculate ppm scale
        ppm_scale = get_ppm_scale(npoints, sw, ori, obs)

        self.axes[dim]["scale"] = ppm_scale
        self.axes[dim]["unit"] = "ppm"

        return self