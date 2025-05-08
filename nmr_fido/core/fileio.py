import numpy as np

from nmr_fido import NMRData

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
    
    print(filepath_or_pattern)
    
    data = NMRData(np.zeros((100, 100)))
    
    with open(filepath_or_pattern, 'rb') as file:
        file_bytes = list(file.read()) # raw bytes
        
    print(file_bytes)
    
    return data


if __name__ == "__main__":
    read_nmrpipe("tests/test.fid")
    
    pass