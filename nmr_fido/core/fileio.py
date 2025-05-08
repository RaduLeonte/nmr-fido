import numpy as np
import nmrglue as ng

from nmr_fido import NMRData


def _decode_bytes(bytes: list | np.ndarray) -> str:
    if isinstance(bytes, list):
        bytes = np.array(bytes)
        
    byte_string = bytes.tobytes()
    
    decoded_string = byte_string.decode('ascii', errors='ignore')
    decoded_string = decoded_string.strip('\x00')
    
    return decoded_string


def _parse_nmrpipe_header(
    header_bytes: np.ndarray
) -> None:
    print(np.array2string(header_bytes, max_line_width=80, precision=3, threshold=5))
    
    # Get nr of dimensions
    ndim = int(header_bytes[9])
    print(ndim)
    
    
    # Get axes labels
    label_bytes_indices = [22, 20, 18, 16] # dim W, Z, Y, X
    labels = [_decode_bytes(header_bytes[i:i+2]) for i in label_bytes_indices[-ndim:]]
    print(labels)
    
    axes = []
    for i in range(ndim):
        axes.append(
            {
                "label": labels[i]
            }
        )
    
    nmrdata_header = NMRData(np.zeros, axes=axes, metadata=None)
    
    return nmrdata_header


def read_nmrpipe(
    filepath_or_pattern
    #*,
) -> NMRData:
    """_summary_

    Args:
        filepath_or_pattern (_type_): _description_

    Returns:
        NMRData: _description_
    """
    
    file_bytes = np.fromfile(filepath_or_pattern, dtype=np.float32)
    header_bytes = file_bytes[:512]
    raw_data = file_bytes[512:]
    
    nmrdata_header = _parse_nmrpipe_header(header_bytes)
    
    
    target_shape = (332, 1500)
    data = raw_data.reshape(*target_shape[:-1], target_shape[-1]*2)
    
    data_is_separated = True
    if data_is_separated:
        # Data is composed of [re1, re2 ... reN, im1, im2 ... imN]
        # reshape into [re1 + 1j*im1, re2 + 1j*im2 ... reN + 1j*imN]
        data = np.array(
            data[..., :target_shape[-1]] + 1j*data[..., target_shape[-1]:],
            dtype=np.complex64
        )

    result = NMRData(data)
    
    return result




if __name__ == "__main__":
    data = read_nmrpipe("tests/test.fid")
    
    print(data)
    pass