# CPC Algorithm

This repository contains the implementation of various causal discovery algorithms, including CPC (our algorithm), FCI (Fast Causal Inference), and kPC (PC with up to separating set of size k).

## Overview

This package implements several causal discovery algorithms:
- CPC (PC-like constraint-based discovery using a conditionally-closed set)
- FCI (Fast Causal Inference)
- kPC (PC-like constraint-based discovery with up to separating set of size k)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from algorithms.core.cpc import CPC
from algorithms.utils.graph_utils import visualize_graph_color

# Initialize and run CPC algorithm
n_rows = 100
n_cols = 10

# Generate random binary data (0 or 1)
data = np.random.randint(0, 2, size=(n_rows, n_cols))

# Create column names: X0, X1, ..., X9
columns = [f'X{i}' for i in range(n_cols)]

# Create DataFrame
df_binary = pd.DataFrame(data, columns=columns)

I = [{}, {'X0'}]
tester = 'chisq'
D, _ = CPC(df_binary.to_numpy(), tester, I, alpha=0.05, data_names=columns)
cpc_adj = D.graph

# Visualize the resulting graph
visualize_graph_color(D, name= 'cpc_output_p{}'.format(alpha)) 
```

## Project Structure

```
cpc/
├── algorithms/           # Main package directory
│   ├── core/            # Core algorithms
│   ├── utils/           # Utility functions
│   └── visualization/   # Graph visualization tools
├── tests/               # Test suite
├── docs/                # Documentation
```

## Documentation

Detailed documentation is available in the `docs/` directory.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

Lee, K., Ribeiro, B., & Kocaoglu, M. Constraint-based Causal Discovery from a Collection of Conditioning Sets.  UAI 2025.

## Contact

Please feel free to open an issue if you have any questions.