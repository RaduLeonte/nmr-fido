from __future__ import annotations
from typing import Callable
import numpy as np
import copy


class NMRData(np.ndarray):
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
            return result  # If numpy returns a scalar, we don't need to modify it.

        if isinstance(item, tuple):
            slicers = list(item) + [slice(None)] * (self.ndim - len(item))
        else:
            slicers = [item] + [slice(None)] * (self.ndim - 1)

        surviving_dims = [i for i, s in enumerate(slicers) if not isinstance(s, int)]

        # Update attributes
        for attr in self._custom_attrs:
            match attr:
                case 'scales':
                    new_scales = []
                    for i, s in enumerate(slicers):
                        if isinstance(s, slice):
                            new_scales.append(self.scales[i][s])
                        elif isinstance(s, int):
                            pass  # Axis squeezed
                        else:
                            new_scales.append(self.scales[i][s])
                    setattr(result, 'scales', new_scales)
                
                case 'labels' | 'units' | 'axis_info':
                    setattr(result, attr, [getattr(self, attr)[i] for i in surviving_dims])
                
                case _:
                    setattr(result, attr, copy.deepcopy(getattr(self, attr)))


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
                lines.append(f"  - {label}: Size={scale.size}, Range=({range_str}), Unit={unit}")

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

        def method(self, overwrite: bool = False, **kwargs):
            result = func(self, **kwargs)
            if overwrite:
                self._update_from(result)
                return self
            else:
                return result

        method.__name__ = name
        setattr(cls, name, method)
        
    
    def scale_to_ppm(self, target_dim: int = -1) -> NMRData:
        """
        Convert the scale of the target dimension to ppm using axis_info.

        Args:
            target_dim (int): Which dimension to convert. Defaults to last dimension (-1).

        Returns:
            NMRData: The updated object (self).
        """
        # Handle negative indices
        dim = target_dim if target_dim >= 0 else self.ndim + target_dim

        if not hasattr(self, 'axis_info') or len(self.axis_info) <= dim:
            raise ValueError(f"No axis_info available for dimension {dim}.")

        info = self.axis_info[dim]

        try:
            sw = info["SW"]   # Sweep Width (Hz)
            sf = info["SF"]   # Spectrometer Frequency (MHz)
            offset = info.get("OFFSET", 0.0)  # Reference ppm, default 0
        except KeyError as e:
            raise ValueError(f"Missing required key in axis_info[{dim}]: {e}")

        npoints = self.shape[dim]

        # Calculate ppm scale
        ppm_scale = np.linspace(
            offset + (sw / (2 * sf)),
            offset - (sw / (2 * sf)),
            npoints,
            endpoint=False
        )

        self.scales[dim] = ppm_scale
        self.units[dim] = "ppm"

        return self


    
from nmr_fido.core import processing

for name in dir(processing):
    attr = getattr(processing, name)
    if callable(attr) and not name.startswith("_"):
        NMRData._add_processing_method(attr)