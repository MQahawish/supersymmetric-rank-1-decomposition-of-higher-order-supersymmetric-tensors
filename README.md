# Supersymmetric-rank-1-decomposition-of-higher-order-supersymmetric-tensors
The main purpose of the script is to decompose suprsymmetric tensors into their rank-1 components. This is achieved by defining a loss function measuring the residual between the original tensor and its reconstruction from the estimated rank-1 components, and applying an optimization algorithm (projected gradient descent here) to minimize the loss.

## Features

- Generate supersymmetric tensors randomly.
- Perform rank-1 supersymmetric tensor decomposition.
- Implement the projected gradient method to find the maximum eigenvalue and corresponding eigenvector of a tensor.
- Visualization of residual norms over iterative steps.
- Command line interaction for tensor dimension, order, and optimization parameters.

## Requirements

The script requires the following libraries:
- `torch` (CUDA compiled)
- `numpy` 
- `matplotlib` 
- `logging`

Ensure you have a Python environment with these packages installed. You can install them using pip:

```bash
pip install torch numpy matplotlib
```

## Usage

Run the script directly from the command line:

```bash
python algorithm.py
```

Follow the on-screen prompts to input parameters such as tensor dimension, order, and optimization settings. Results will be displayed directly in the console, and plots will illustrate the change in Frobenius norms of the residuals.

## Functions Overview

### `random_supersymmetric_tensor`
Generates a random supersymmetric tensor of specified dimension and order.

### `rank1_supersymmetric_tensor`
Computes a rank-1 supersymmetric tensor based on a given vector.

### `sum_rank1_supersymmetric_tensors`
Sums a list of rank-1 tensors each scaled by a corresponding scalar.

### `successive_rank1_decomp`
performs a successive rank-1 decomposition of a supersymmetric tensor.

### `projected_gradient_method`
Applies the projected gradient method for optimization on tensor spaces. (Good start is alpha=[0.01,0.1] , beta=[0.4,0.8])

### `generate_grid_points`
Generates random points on the unit sphere for initial guesses in optimization algorithms.

### `plot_metrics`
Plots the Frobenius norm of residual tensors as a function of iteration steps in the decomposition process.

## Citation
based on the paper : https://www.polyu.edu.hk/ama/staff/new/qilq/WQ.pdf

## License

MIT
