from __future__ import annotations
import numpy as np
import copy

class NMRData(np.ndarray):
    _custom_attrs = ['labels', 'scales', 'units', 'metadata', 'processing_history']
    
    def __new__(
        cls,
        input_array: np.ndarray,
        labels: list[str] = None,
        scales: list[np.ndarray] = None,
        units: list[str] = None,
        metadata: dict = None,
        processing_history: list[dict] = None,
        copy_from: NMRData = None,
    ):
        obj: np.ndarray = np.asarray(input_array).view(cls)
        
        
        init_args = {
            'labels': labels,
            'scales': scales,
            'units': units,
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
                
                case 'labels' | 'units':
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
    
    
    def transpose(self, *axes) -> NMRData:
        # Transpose data
        result = super().transpose(*axes)
        
        # Reorder axes
        if axes:
            axes = axes[0]
        else:
            axes = reversed(range(self.ndim))

        # Update relevant attributes
        for attr in self._custom_attrs:
            match attr:
                case 'scales' | 'labels' | 'units':
                    # Reorder attribute
                    setattr(result, attr, [getattr(self, attr)[ax] for ax in axes])

                case _:
                    # Deep copy attribute
                    setattr(result, attr, copy.deepcopy(getattr(self, attr)))

        return result
