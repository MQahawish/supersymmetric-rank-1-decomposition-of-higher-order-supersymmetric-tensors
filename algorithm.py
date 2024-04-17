import torch
import numpy as np
import math
from itertools import permutations
import logging
import matplotlib.pyplot as plt
import json

logging.basicConfig(level=logging.INFO,
                 format='%(asctime)s - %(levelname)s - %(message)s'
                 )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16  

def random_supersymmetric_tensor(dimension, order, device='cpu'):
    shape = (dimension,) * order
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    
    indices_list = torch.combinations(torch.arange(dimension), order, with_replacement=True)
    
    for indices in indices_list:
        value = torch.rand(1, dtype=dtype, device=device) 
        for perm in permutations(indices):
            tensor[perm] = value

    return tensor

def rank1_supersymmetric_tensor(vector, order):
    tensor = vector
    tensor=tensor.to(dtype)
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

def projected_gradient_method(tensor, tol, alpha, beta, max_iter=100, grid_size=1000, device='cpu'):
    tensor = tensor.to(dtype)
    dimension = tensor.shape[0]
    order = tensor.ndim
    logging.info(f"Starting projected gradient method with tensor of shape {tensor.shape}")

    grid_points = generate_grid_points(dimension, grid_size, device)
    logging.info(f"Generated {grid_points.shape[0]} grid points on the unit sphere")

    vector = select_initial_vector(tensor, grid_points)
    logging.info(f"Initial vector = {vector}, Norm: {torch.norm(vector)}")

    prev_vector = vector.clone()
    prev_obj_value = compute_objective(tensor, prev_vector)

    i = 0
    while i < max_iter:
        gradient = compute_gradient(tensor, vector, order)
        projected_gradient = project_gradient(gradient)
        stepsize = armijo_stepsize_rule(tensor, vector, projected_gradient, alpha, beta)
        vector = update_vector(vector, stepsize, projected_gradient)

        obj_value = compute_objective(tensor, vector)
        logging.info(f"~~~~~~Iteration {i+1}: Stepsize = {stepsize}, Objective Value = {obj_value}")

        if torch.norm(vector - prev_vector) < tol or abs(obj_value - prev_obj_value) < tol:
            logging.info(f"Converged after {i+1} iterations")
            break

        prev_vector = vector.clone()
        prev_obj_value = obj_value.clone()
        i += 1

    if i == max_iter:
        logging.info(f"Reached maximum iterations ({max_iter})")

    lamb = compute_lambda(tensor, vector, order)
    logging.info(f"Projected gradient method completed, final lambda = {lamb}")
    return lamb, vector

def generate_grid_points(dimension, grid_size, device):
    random_points = torch.randn(grid_size, dimension, device=device)
    random_points /= torch.norm(random_points,p=2, dim=1, keepdim=True)
    return random_points

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
    outer_product = torch.outer(vector, vector).to(tensor.dtype)
    tensor_shape = [tensor.shape[0]] * tensor.ndim
    outer_product_reshaped = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    for idx in np.ndindex(*tensor_shape):
        outer_product_reshaped[idx] = outer_product[idx[-2], idx[-1]]
    return torch.tensordot(tensor, outer_product_reshaped, dims=tensor.ndim)

def compute_lambda(tensor, vector, order):
    outer_product = torch.outer(vector, vector).to(tensor.dtype)
    tensor_shape = [tensor.shape[0]] * tensor.ndim
    outer_product_reshaped = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    for idx in np.ndindex(*tensor_shape):
        outer_product_reshaped[idx] = outer_product[idx[-2], idx[-1]]
    return torch.tensordot(tensor, outer_product_reshaped, dims=tensor.ndim)

def successive_rank1_decomp(tensor, max_iter=100, tol=1e-6, alpha=0.01, beta=0.9, device='cpu'):
    tensor = tensor.to(dtype)
    tensor_res = tensor.clone().to(device)
    lambs = []
    vectors = []
    residual_norms = []
    logging.info(f"Starting successive rank-1 decomposition with tensor of shape {tensor.shape}")
    for i in range(max_iter):
        lamb, vector = projected_gradient_method(tensor_res, tol, alpha, beta, device=device)
        lambs.append(lamb.cpu())
        vectors.append(vector.cpu())

        rank1_update = lamb * rank1_supersymmetric_tensor(vector, tensor.ndim).to(device)
        tensor_res -= rank1_update

        residual_norm = torch.norm(tensor_res).item()  
        residual_norms.append(residual_norm) 
        logging.info(f"Residual tensor norm = {residual_norm} , Lambda = {lamb}, Vector = {vector}")

        if residual_norm < tol:
            logging.info(f"Decomposition converged after {i+1} iterations")
            break
    logging.info("Successive rank-1 decomposition completed")

    return lambs, vectors, residual_norms

def collect_integer(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

def collect_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def generate_tensor():
    dimension = collect_integer("Dimension: ")
    order = collect_integer("Order: ")
    tensor = random_supersymmetric_tensor(dimension, order, device='cpu')
    return tensor, order

def perform_decomposition(tensor, device):
    max_iter = collect_integer("Enter max iterations: ")
    tol = collect_float("Enter tolerance: ")
    alpha = collect_float("Enter alpha: ")
    beta = collect_float("Enter beta: ")
    return successive_rank1_decomp(tensor, max_iter=max_iter, tol=tol, alpha=alpha, beta=beta, device=device)

def plot_metrics(residual_norms):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(residual_norms) + 1), residual_norms, marker='o')
    plt.xlabel('Iteration Steps')
    plt.ylabel('Frobenius Norm of Residual Tensors')
    plt.title('Frobenius Norm of Residual Tensors vs. Iteration Steps')
    plt.grid(True)
    plt.show()

def main():
    while True:
        print("Generate random tensor(Dimension + Order)")
        tensor, order = generate_tensor()
        print("Original Tensor:")
        print(tensor)
        print("Original tensor shape:", tensor.shape)

        lambs, vectors, residual_norms = perform_decomposition(tensor, 'cpu')

        print("\nDecomposition Results:")
        for i, (lamb, vector) in enumerate(zip(lambs, vectors)):
            print(f"Rank-1 Tensor {i+1}:")
            print(f"Lambda: {lamb}")
            print(f"Vector: {vector}")
            print()

        tensor_recon = sum_rank1_supersymmetric_tensors(zip(lambs, vectors), order).to('cpu')
        print("Original Tensor:")
        print(tensor)
        print("Reconstructed Tensor:")
        print(tensor_recon)
        error = torch.norm(tensor - tensor_recon).item()
        print(f"\nReconstruction Error: {error}")

        plot_metrics(residual_norms)

if __name__ == '__main__':
    main()