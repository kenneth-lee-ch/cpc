import pytest
import numpy as np
from algorithms.core.cpc import CPC

def test_cpc_initialization():
    """Test CPC initialization with valid parameters."""
    data = np.random.randn(100, 5)  # 100 samples, 5 variables
    cpc = CPC(data, alpha=0.05)
    assert cpc.alpha == 0.05
    assert cpc.data.shape == (100, 5)

def test_cpc_invalid_data():
    """Test CPC initialization with invalid data."""
    with pytest.raises(ValueError):
        CPC(None, alpha=0.05)
    
    with pytest.raises(ValueError):
        CPC(np.array([]), alpha=0.05)

def test_cpc_invalid_alpha():
    """Test CPC initialization with invalid alpha values."""
    data = np.random.randn(100, 5)
    with pytest.raises(ValueError):
        CPC(data, alpha=-0.1)
    
    with pytest.raises(ValueError):
        CPC(data, alpha=1.1)

# Add more tests as needed 