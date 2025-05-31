"""Tests for conditional independence tests."""

import pytest
import numpy as np
from algorithms.utils.cit import (
    compute_probabilities,
    mutual_information,
    conditional_mutual_information
)

def test_compute_probabilities():
    """Test probability computation for discrete data."""
    # Test case 1: Simple discrete data
    data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    probs = compute_probabilities(data)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.isclose(np.sum(probs), 1.0)

    # Test case 2: Different sampling size
    probs = compute_probabilities(data, sampling_size=2)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.isclose(np.sum(probs), 1.0)

def test_mutual_information():
    """Test mutual information calculation."""
    # Test case 1: Independent variables
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    mi = mutual_information(x, y)
    assert mi >= 0
    assert mi <= 1

    # Test case 2: Perfectly dependent variables
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 1])
    mi = mutual_information(x, y)
    assert mi > 0

def test_conditional_mutual_information():
    """Test conditional mutual information calculation."""
    # Test case 1: Independent variables with conditioning set
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    z = np.array([0, 0, 1, 1])
    cmi = conditional_mutual_information(x, y, z)
    assert cmi >= 0

    # Test case 2: Empty conditioning set
    cmi = conditional_mutual_information(x, y, np.array([]))
    assert cmi >= 0

    # Test case 3: None conditioning set
    cmi = conditional_mutual_information(x, y, None)
    assert cmi >= 0 