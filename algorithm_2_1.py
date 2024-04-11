import torch
import numpy as np
import math
from itertools import permutations
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)  # Configure logging to display INFO level messages

# If cuda is available, device will be set to cuda. Otherwise, it will be set to cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_supersymmetric_tensor(dimension, order, device='cpu'):
    shape = (dimension,) * order
    tensor = torch.rand(shape, device=device, dtype=torch.float16)
    result = tensor.clone()
    for perm in permutations(range(order)):
        result += tensor.permute(perm)
    
    result /= math.factorial(order)
    return result

def rank1_supersymmetric_tensor(vector, order):
    tensor = vector
    tensor=tensor.to(torch.float16)
    for _ in range(order - 1):
        tensor = torch.tensordot(tensor, vector, dims=0)
    return tensor

def sum_rank1_supersymmetric_tensors(lambs_vectors, order):
    lambs, vectors = zip(*lambs_vectors)
    vector = vectors[0]
    reconstructed_tensor = torch.zeros_like(rank1_supersymmetric_tensor(vector, order))
    for lamb, vector in zip(lambs, vectors):
        rank1_tensor = rank1_supersymmetric_tensor(vector, order)
        reconstructed_tensor += lamb * rank1_tensor
    return reconstructed_tensor

def projected_gradient_method(tensor, max_iter=100, tol=1e-6, alpha=0.01, beta=0.9, grid_size=10, device='cpu'):
    tensor = tensor.to(torch.float16)
    dimension = tensor.shape[0]
    order = tensor.ndim
    logging.info(f"Starting projected gradient method with tensor of shape {tensor.shape}")

    # Generate a uniform grid of points on the unit sphere
    grid_points = generate_grid_points(dimension, grid_size, device)
    logging.info(f"Generated {grid_points.shape[0]} grid points on the unit sphere")

    # Evaluate the tensor at each grid point and select the best one as the initial vector
    vector = select_initial_vector(tensor, grid_points)

    logging.info(f"Initial vector = {vector}, Norm: {torch.norm(vector)}")

    prev_stepsize = None
    early_stopping_counter = 0
    early_stopping_threshold = max_iter // 2

    prev_vector = vector.clone()

    for i in range(max_iter):
        gradient = compute_gradient(tensor, vector, order)
        projected_gradient = project_gradient(gradient)
        stepsize = armijo_stepsize_rule(tensor, vector, projected_gradient, alpha, beta)
        vector = update_vector(vector, stepsize, projected_gradient)

        logging.info(f"Iteration {i+1}: Stepsize = {stepsize}")

        if torch.norm(vector - prev_vector) < tol:
            logging.info(f"Converged after {i+1} iterations")
            break

        if prev_stepsize == stepsize:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        prev_stepsize = stepsize
        prev_vector = vector.clone()

        if early_stopping_counter >= early_stopping_threshold:
            logging.info("Early stopping criteria met, exiting iteration")
            break

    lamb = compute_lambda(tensor, vector, order)
    logging.info(f"Projected gradient method completed, final lambda = {lamb}")
    return lamb, vector

def generate_grid_points(dimension, grid_size, device):
    grid = torch.meshgrid(*[torch.linspace(-1, 1, grid_size, device=device) for _ in range(dimension)], indexing='ij')
    grid_points = torch.stack([grid[i].flatten() for i in range(dimension)]).T
    grid_points /= torch.norm(grid_points, dim=1).unsqueeze(1)
    return grid_points

def select_initial_vector(tensor, grid_points):
    tensor_values = compute_tensor_values(tensor, grid_points)
    best_index = torch.argmin(tensor_values)
    vector = grid_points[best_index].to(tensor.dtype)
    return vector

def compute_tensor_values(tensor, grid_points):
    tensor_values = torch.zeros(grid_points.shape[0], device=tensor.device, dtype=tensor.dtype)
    for i in range(grid_points.shape[0]):
        grid_point_cubed = grid_points[i].pow(tensor.ndim).unsqueeze(0)
        tensor_values[i] = -1*torch.sum(tensor * grid_point_cubed)
    return tensor_values

def compute_gradient(tensor, vector, order):
    gradient = tensor.clone()
    for _ in range(order - 1):
        gradient = torch.tensordot(gradient, vector, dims=1)
    return gradient

def project_gradient(gradient):
    return gradient / max(torch.norm(gradient), 1)

def armijo_stepsize_rule(tensor, vector, projected_gradient, alpha, beta):
    stepsize = 1.0
    min_stepsize = 1e-8
    while stepsize > min_stepsize:
        new_vector = update_vector(vector, stepsize, projected_gradient)
        new_obj = compute_objective(tensor, new_vector)
        old_obj = compute_objective(tensor, vector)
        if new_obj >= old_obj + alpha * stepsize * torch.dot(projected_gradient, vector - new_vector):
            break
        stepsize *= beta
    return stepsize

def update_vector(vector, stepsize, projected_gradient):
    new_vector = vector - stepsize * projected_gradient
    new_vector /= torch.norm(new_vector)
    return new_vector

def compute_objective(tensor, vector):
    tensor_shape = [tensor.shape[0]] * tensor.ndim
    tensor_shape[-1] = 1
    return torch.tensordot(tensor, torch.outer(vector, vector).to(tensor.dtype).reshape(tensor_shape), dims=tensor.ndim)

def compute_lambda(tensor, vector, order):
    tensor_shape = [tensor.shape[0]] * order
    tensor_shape[-1] = 1
    return torch.tensordot(tensor, torch.outer(vector, vector).to(tensor.dtype).reshape(tensor_shape), dims=order)

def successive_rank1_decomp(tensor, max_iter=100, tol=1e-6, alpha=0.01, beta=0.9, device='cpu'):
    tensor = tensor.to(torch.float16)
    tensor_res = tensor.clone().to(device)
    lambs = []
    vectors = []

    logging.info(f"Starting successive rank-1 decomposition with tensor of shape {tensor.shape}")
    for i in range(max_iter):
        logging.info(f"Decomposition iteration {i+1}")
        lamb, vector = projected_gradient_method(tensor_res, max_iter*10, tol, alpha, beta, device=device)
        lambs.append(lamb.cpu())
        vectors.append(vector.cpu())

        rank1_update = lamb * rank1_supersymmetric_tensor(vector, tensor.ndim).to(device)
        tensor_res -= rank1_update

        logging.info(f"Rank-1 tensor {i+1}: Lambda = {lamb}, Vector = {vector}")
        logging.info(f"Residual tensor norm = {torch.norm(tensor_res)}")

        if torch.norm(tensor_res) < tol:
            logging.info(f"Decomposition converged after {i+1} iterations")
            break
    logging.info("Successive rank-1 decomposition completed")

    return lambs, vectors


def main():
    # Example usage
    dimension = 2
    order = 3
    tensor = random_supersymmetric_tensor(dimension, order, device=device)  # Create tensor on GPU
    errors = []
    max_iter_values=[]
    for max_iter in range(10, 50):
        lambs, vectors = successive_rank1_decomp(tensor,max_iter=max_iter,tol=1e-8, alpha=0.01, beta=0.5, device=device)  # Perform decomposition on GPU
        max_iter_values.append(max_iter)
        print("Original Tensor:")
        print(tensor)
        print("Origiinal tensor shape:", tensor.shape)

        print("\nDecomposition Results:")
        for i, (lamb, vector) in enumerate(zip(lambs, vectors)):
            print(f"Rank-1 Tensor {i+1}:")
            print(f"Lambda: {lamb}")
            print(f"Vector: {vector}")
            print()

        tensor_recon = sum_rank1_supersymmetric_tensors(zip(lambs, vectors), order).to(device) # Reconstruct tensor on GPU
        print("Reconstructed Tensor:")
        print(tensor_recon)
        print("Reconstructed tensor shape:", tensor_recon.shape)

        error = torch.norm(tensor - tensor_recon).item()
        print(f"\nReconstruction Error: {error}")
        errors.append(error)
        # Plot the error vs. max_iter
    plt.figure(figsize=(8, 6))
    plt.plot(max_iter_values, errors, marker='o')
    plt.xlabel('Max Iterations')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs. Max Iterations')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()