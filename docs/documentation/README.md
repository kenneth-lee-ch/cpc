# UAI CPC Code Documentation

## Overview

This documentation provides detailed information about the CPC Code package, its components, and usage.

## Core Components

### CPC (paper reference: Lee, Kenneth, Bruno Ribeiro, and Murat Kocaoglu. "Constraint-based Causal Discovery from a Collection of Conditioning Sets." 9th Causal Inference Workshop at UAI 2024.)
The CPC algorithm is implemented in `algorithms/core/cpc.py`. 

### FCI (Fast Causal Inference)
The FCI algorithm is implemented in `algorithms/core/fci.py`. It extends the PC algorithm to handle latent variables.

### kPC (paper reference: Kocaoglu, Murat. "Characterization and learning of causal graphs with small conditioning sets." Advances in Neural Information Processing Systems 36 (2023): 74140-74179.)
The kPC algorithm is implemented in `algorithms/core/kpc.py`. 

## Utility Functions

### Graph Utilities
Located in `algorithms/utils/graph_utils.py`, these functions provide various graph manipulation and analysis tools.

### Conditional Independence Tests
Located in `algorithms/utils/cit.py`, these functions implement various conditional independence tests.

## Visualization

Graph visualization tools are available in the visualization module, providing various ways to display and analyze causal graphs.


## API Reference

### CPC Class
```python
class CPC:
    def __init__(self, data: np.ndarray, alpha: float = 0.05):
        """
        Initialize the CPC algorithm.
        
        Args:
            data: Input data matrix
            alpha: Significance level for independence tests
        """
```

### FCI Class
```python
class FCI:
    def __init__(self, data: np.ndarray, alpha: float = 0.05):
        """
        Initialize the FCI algorithm.
        
        Args:
            data: Input data matrix
            alpha: Significance level for independence tests
        """
```

## Contributing

Please refer to the main README.md for contribution guidelines.

## License

This project is licensed under the MIT License. 