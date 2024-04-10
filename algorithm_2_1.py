import torch
import numpy as np
import math
from itertools import permutations
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging to display INFO level messages

# If cuda is available, device will be set to cuda. Otherwise, it will be set to cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_supersymmetric_tensor(dimension, order, device='cpu'):
    shape = (dimension,) * order
    tensor = torch.rand(shape, device=device)
    
    result = tensor.clone()
    for perm in permutations(range(order)):
        result += tensor.permute(perm)
    
    result /= math.factorial(order)
    return result

def rank1_supersymmetric_tensor(vector, order):
    tensor = vector
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

def projected_gradient_method(tensor, max_iter=100, tol=1e-6, alpha=0.01, beta=0.9, grid_size=30, device='cpu'):
    dimension = tensor.shape[0]
    order = tensor.ndim

    logging.info(f"Starting projected gradient method with tensor of shape {tensor.shape}")

    # Generate a uniform grid of points on the unit sphere
    grid = torch.meshgrid(*[torch.linspace(-1, 1, grid_size, device=device) for _ in range(dimension)], indexing='ij')
    grid_points = torch.stack([grid[i].flatten() for i in range(dimension)]).T
    grid_points /= torch.norm(grid_points, dim=1).unsqueeze(1)

    logging.info(f"Generated {grid_points.shape[0]} grid points on the unit sphere")

    # Evaluate the tensor at each grid point and select the best one as the initial vector
    tensor_values = torch.zeros(grid_size**dimension, device=device)
    for i in range(grid_points.shape[0]):
        outer_product = torch.outer(grid_points[i], grid_points[i])
        tensor_shape = [dimension] * order
        tensor_shape[-1] = 1
        reshaped_outer_product = outer_product.reshape(tensor_shape)
        tensor_values[i] = torch.tensordot(tensor, reshaped_outer_product, dims=order)
    best_index = torch.argmax(torch.abs(tensor_values))
    vector = grid_points[best_index]

    prev_stepsize = None
    prev_gradient_norm = None
    early_stopping_counter = 0
    early_stopping_threashold = 40
    for i in range(max_iter):
        gradient = tensor.clone()
        for _ in range(order - 1):
            gradient = torch.tensordot(gradient, vector, dims=1)
        # Projection onto the unit ball
        projected_gradient = gradient / max(torch.norm(gradient), 1)

        # Armijo stepsize rule
        stepsize = 1.0
        min_stepsize = 0  # Define a minimum step size

        while stepsize > min_stepsize:
            new_vector = vector - stepsize * projected_gradient
            new_vector /= torch.norm(new_vector)

            tensor_shape = [dimension] * order
            tensor_shape[-1] = 1
            
            # Calculate the new objective
            new_obj = torch.tensordot(tensor, torch.outer(new_vector, new_vector).reshape(tensor_shape), dims=order)
            old_obj = torch.tensordot(tensor, torch.outer(vector, vector).reshape(tensor_shape), dims=order)
            
            if new_obj >= old_obj + alpha * stepsize * torch.dot(gradient, vector - new_vector):
                break  # If the condition is met, exit the loop
            
            stepsize *= beta  # Reduce stepsize

            # if stepsize <= min_stepsize:
            #     logging.warning("Stepsize reached the minimum threshold, exiting loop to prevent getting stuck.")
            #     break

        vector = new_vector
        gradient_norm = torch.norm(projected_gradient)
        logging.info(f"Iteration {i+1}: Stepsize = {stepsize}, Gradient norm = {gradient_norm}")

        # if prev_stepsize != stepsize or prev_gradient_norm != gradient_norm:
        #     logging.info(f"Iteration {i+1}: Stepsize = {stepsize}, Gradient norm = {gradient_norm}")
        #     prev_stepsize = stepsize
        #     prev_gradient_norm = gradient_norm
        # else:
        #     early_stopping_counter += 1

        if gradient_norm < tol:
            logging.info(f"Converged after {i+1} iterations")
            break
        if early_stopping_counter >= early_stopping_threashold:
            logging.info("Early stopping criteria met, exiting iteration")
            break

    tensor_shape = [dimension] * order
    tensor_shape[-1] = 1
    lamb = torch.tensordot(tensor, torch.outer(vector, vector).reshape(tensor_shape), dims=order)
    logging.info(f"Projected gradient method completed, final lambda = {lamb}")
    return lamb, vector

def successive_rank1_decomp(tensor, max_iter=50, tol=1e-6, alpha=0.1, beta=0.5, device='cpu'):
    tensor_res = tensor.clone().to(device)
    lambs = []
    vectors = []

    logging.info(f"Starting successive rank-1 decomposition with tensor of shape {tensor.shape}")

    for i in range(max_iter):
        logging.info(f"Decomposition iteration {i+1}")

        lamb, vector = projected_gradient_method(tensor_res, max_iter, tol, alpha, beta, device=device)
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

# Example usage
dimension = 3
order = 3
tensor = random_supersymmetric_tensor(dimension, order, device=device)  # Create tensor on GPU

lambs, vectors = successive_rank1_decomp(tensor, device=device)  # Perform decomposition on GPU

print("Original Tensor:")
print(tensor)
print("Origiinal tensor shape:", tensor.shape)

print("\nDecomposition Results:")
for i, (lamb, vector) in enumerate(zip(lambs, vectors)):
    print(f"Rank-1 Tensor {i+1}:")
    print(f"Lambda: {lamb}")
    print(f"Vector: {vector}")
    print()

tensor_recon = sum_rank1_supersymmetric_tensors(zip(lambs, vectors), order).to(device)  # Reconstruct tensor on GPU
print("Reconstructed Tensor:")
print(tensor_recon)
print("Reconstructed tensor shape:", tensor_recon.shape)

error = torch.norm(tensor - tensor_recon).item()
print(f"\nReconstruction Error: {error}")