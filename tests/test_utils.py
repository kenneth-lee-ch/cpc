"""Tests for utility functions."""

import pytest
import numpy as np
import pandas as pd
from algorithms.utils.graph_utils import (
    fscore_calculator_skeleton,
    fscore_calculator_arrowhead,
    fscore_calculator_tail
)

def test_fscore_calculator_skeleton():
    """Test skeleton F-score calculation."""
    # Test case 1: Perfect match
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    true_adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    assert fscore_calculator_skeleton(adj, true_adj) == 1.0

    # Test case 2: No match
    adj = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    true_adj = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ])
    assert fscore_calculator_skeleton(adj, true_adj) == 0.0

def test_fscore_calculator_arrowhead():
    """Test arrowhead F-score calculation."""
    # Test case 1: Perfect match
    adj = np.array([
        [0, 1, 0],
        [-1, 0, 1],
        [0, -1, 0]
    ])
    true_adj = np.array([
        [0, 1, 0],
        [-1, 0, 1],
        [0, -1, 0]
    ])
    assert fscore_calculator_arrowhead(adj, true_adj) == 1.0

    # Test case 2: No match
    adj = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ])
    true_adj = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [-1, -1, 0]
    ])
    assert fscore_calculator_arrowhead(adj, true_adj) == 0.0

def test_fscore_calculator_tail():
    """Test tail F-score calculation."""
    # Test case 1: Perfect match
    adj = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0]
    ])
    true_adj = np.array([
        [0, -1, 0],
        [1, 0, -1],
        [0, 1, 0]
    ])
    assert fscore_calculator_tail(adj, true_adj) == 1.0

    # Test case 2: No match
    adj = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    true_adj = np.array([
        [0, 0, -1],
        [0, 0, -1],
        [1, 1, 0]
    ])
    assert fscore_calculator_tail(adj, true_adj) == 0.0 