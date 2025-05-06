import numpy as np
import pytest
import nmr_fido as nf

@pytest.fixture
def sample_data():
    array = np.ones((4, 8), dtype=np.complex128)
    return nf.NMRData(array)

def test_sine_bell_window_runs(sample_data):
    result = nf.sine_bell_window(sample_data)
    assert isinstance(result, nf.NMRData)
    assert result.shape == sample_data.shape

def test_zero_fill_factor(sample_data):
    result = nf.zero_fill(sample_data, factor=1)
    assert isinstance(result, nf.NMRData)
    assert result.shape[-1] == sample_data.shape[-1] * 2

def test_zero_fill_final_size(sample_data):
    result = nf.zero_fill(sample_data, final_size=20)
    assert result.shape[-1] == 20

def test_fourier_transform_basic(sample_data):
    result = nf.fourier_transform(sample_data, real_only=True)
    assert isinstance(result, nf.NMRData)
    assert result.shape == sample_data.shape
