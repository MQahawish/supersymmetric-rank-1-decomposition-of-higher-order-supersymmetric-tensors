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
# If cuda is available, device will be set to cuda. Otherwise, it will be set to cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64  # Define the desired dtype for tensors

def load_tensor_from_json(file_path,device):
    # Load the JSON file
    with open(file_path, 'r') as f:
        tensors = json.load(f)

    # Display tensors to the user for selection
    print("Please choose one of the following tensors by entering its number:")
    for i, tensor in enumerate(tensors):
        print(f"{i + 1}: Tensor of order {tensor['order']} with dimensions {tensor['dimensions']}")

    # User input to choose the tensor
    choice = int(input("Enter your choice (number): ")) - 1
    if choice < 0 or choice >= len(tensors):
        print("Invalid choice, exiting.")
        return None, None

    # Extract the chosen tensor's data
    selected_tensor = tensors[choice]
    tensor_order = selected_tensor['order']
    tensor_dimensions = selected_tensor['dimensions'][0]
    tensor_entries = selected_tensor['entries']

    # Create a PyTorch tensor from the entries
    tensor = torch.tensor(tensor_entries, device=device, dtype=dtype)

    return tensor_order, tensor_dimensions, tensor

def random_supersymmetric_tensor(dimension, order, device='cpu'):
    # Initialize an empty tensor of the given shape and type
    shape = (dimension,) * order
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    
    # Generate all possible index combinations for the given dimension and order
    indices_list = torch.combinations(torch.arange(dimension), order, with_replacement=True)
    
    # Assign a random value to each unique set of indices and its permutations
    for indices in indices_list:
        value = torch.rand(1, dtype=dtype, device=device)  # Generate a random value
        # Set this value for all permutations of the current index tuple
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

def projected_gradient_method(tensor, max_iter, tol, alpha, beta, grid_size=100, device='cpu'):
    tensor = tensor.to(dtype)
    dimension = tensor.shape[0]
    order = tensor.ndim
    logging.info(f"Starting projected gradient method with tensor of shape {tensor.shape}")

    # Generate a uniform grid of points on the unit sphere
    grid_points = generate_grid_points(dimension, grid_size, device)
    logging.info(f"Generated {grid_points.shape[0]} grid points on the unit sphere")

    # Evaluate the tensor at each grid point and select the best one as the initial vector
    vector = select_initial_vector(tensor, grid_points)

    logging.info(f"Initial vector = {vector}, Norm: {torch.norm(vector)}")

    prev_vector = vector.clone()

    i = 0
    while True:
            gradient = compute_gradient(tensor, vector, order)
            projected_gradient = project_gradient(gradient)
            stepsize = armijo_stepsize_rule(tensor, vector, projected_gradient, alpha, beta)
            prev_vector = vector.clone()
            vector = update_vector(vector, stepsize, projected_gradient)

            logging.info(f"~~~~~~Iteration {i+1}: Stepsize = {stepsize} , Gradient = {gradient}")

            if torch.norm(vector - prev_vector) < tol:
                logging.info(f"Converged after {i+1} iterations")
                break
            
            i += 1

    lamb = compute_lambda(tensor, vector, order)
    logging.info(f"Projected gradient method completed, final lambda = {lamb}")
    return lamb, vector

def generate_grid_points(dimension, grid_size, device):
    # Generate random points from a normal distribution
    random_points = torch.randn(grid_size, dimension, device=device)
    # Normalize each point to lie on the unit sphere
    random_points /= torch.norm(random_points, dim=1, keepdim=True)
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
    residual_norms = []  # List to store the Frobenius norms of residual tensors

    logging.info(f"Starting successive rank-1 decomposition with tensor of shape {tensor.shape}")
    for i in range(max_iter):
        lamb, vector = projected_gradient_method(tensor_res, max_iter*10, tol, alpha, beta, device=device)
        lambs.append(lamb.cpu())
        vectors.append(vector.cpu())

        rank1_update = lamb * rank1_supersymmetric_tensor(vector, tensor.ndim).to(device)
        tensor_res -= rank1_update

        residual_norm = torch.norm(tensor_res).item()  # Calculate the Frobenius norm of the residual tensor
        residual_norms.append(residual_norm)  # Append the Frobenius norm to the list
        logging.info(f"Residual tensor norm = {residual_norm} , Lambda = {lamb}, Vector = {vector}")

        if residual_norm < tol:
            logging.info(f"Decomposition converged after {i+1} iterations")
            break
    logging.info("Successive rank-1 decomposition completed")

    return lambs, vectors, residual_norms


def main():
    while True:
        # Example usage
        print("Pick an option:")
        print("1. Generate random tensor")
        print("2. Pick tensor from test set")
        option = int(input())
        if option == 1:
            print("Enter dimension and order of tensor:")
            dimension = int(input("Dimension: "))
            order = int(input("Order: "))
            tensor = random_supersymmetric_tensor(dimension, order, device='cpu')
        elif option == 2:
            order, dimension, tensor = load_tensor_from_json('test_tensors.json',device='cpu')
        else:
            print("Invalid option")
            continue
        print("Original Tensor:")
        print(tensor)
        print("Origiinal tensor shape:", tensor.shape)
        errors = []
        max_iter = int(input("Enter max iterations: "))
        tol = float(input("Enter tolerance: "))
        alpha = float(input("Enter alpha: "))
        beta = float(input("Enter beta: "))
        lambs, vectors, residual_norms = successive_rank1_decomp(tensor, max_iter=max_iter, tol=tol, alpha=alpha, beta=beta, device=device)
        print("\nDecomposition Results:")
        for i, (lamb, vector) in enumerate(zip(lambs, vectors)):
            print(f"Rank-1 Tensor {i+1}:")
            print(f"Lambda: {lamb}")
            print(f"Vector: {vector}")
            print()

        tensor_recon = sum_rank1_supersymmetric_tensors(zip(lambs, vectors), order).to('cpu') # Reconstruct tensor on GPU
        print("Original Tensor:")
        print(tensor)
        print("Reconstructed Tensor:")
        print(tensor_recon)

        error = torch.norm(tensor - tensor_recon).item()
        print(f"\nReconstruction Error: {error}")
        errors.append(error)
        # Plot the error vs. max_iter
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(residual_norms) + 1), residual_norms, marker='o')
        plt.xlabel('Iteration Steps')
        plt.ylabel('Frobenius Norm of Residual Tensors')
        plt.title('Frobenius Norm of Residual Tensors vs. Iteration Steps')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()