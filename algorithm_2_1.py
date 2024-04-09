import numpy as np
from itertools import permutations

def random_supersymmetric_tensor(dimension, order):
    shape = (dimension,) * order
    tensor = np.random.rand(*shape)
    
    for perm in permutations(range(order)):
        tensor += tensor.transpose(perm)
    
    tensor /= np.math.factorial(order)
    return tensor

def rank1_supersymmetric_tensor(vector, order):
    return np.tensordot(vector, vector, axes=0) ** (order - 1)

def sum_rank1_supersymmetric_tensors(vectors, order):
    return sum(rank1_supersymmetric_tensor(vector, order) for vector in vectors)

def projected_gradient_method(tensor, max_iter=100, tol=1e-6, alpha=0.1, beta=0.5, grid_size=10):
    dimension = tensor.shape[0]
    order = tensor.ndim

    # Generate a uniform grid of points on the unit sphere
    grid = np.meshgrid(*[np.linspace(-1, 1, grid_size) for _ in range(dimension)])
    grid_points = np.vstack([grid[i].ravel() for i in range(dimension)]).T
    grid_points /= np.linalg.norm(grid_points, axis=1)[:, np.newaxis]

    # Evaluate the tensor at each grid point and select the best one as the initial vector
    tensor_values = np.zeros(grid_size**dimension)
    for i in range(grid_points.shape[0]):
        tensor_values[i] = np.tensordot(tensor, np.outer(grid_points[i], grid_points[i]).reshape(tensor.shape), axes=order)
    best_index = np.argmax(np.abs(tensor_values))
    vector = grid_points[best_index]

    for i in range(max_iter):
        gradient = tensor.copy()
        for _ in range(order - 1):
            gradient = np.tensordot(gradient, vector, axes=1)

        # Projection onto the unit ball
        projected_gradient = gradient / max(np.linalg.norm(gradient), 1)

        # Armijo stepsize rule
        stepsize = 1.0
        while True:
            new_vector = vector - stepsize * projected_gradient
            new_vector /= np.linalg.norm(new_vector)
            if np.tensordot(tensor, np.outer(new_vector, new_vector).reshape(tensor.shape), axes=order) >= np.tensordot(tensor, np.outer(vector, vector).reshape(tensor.shape), axes=order) + alpha * stepsize * np.dot(gradient, vector - new_vector):
                break
            stepsize *= beta

        vector = new_vector

        if np.linalg.norm(projected_gradient) < tol:
            break

    lamb = np.tensordot(tensor, np.outer(vector, vector).reshape(tensor.shape), axes=order)
    return lamb, vector

def successive_rank1_decomp(tensor, max_iter=100, tol=1e-6, alpha=0.1, beta=0.5):
    tensor_res = tensor.copy()
    lambs = []
    vectors = []

    for i in range(max_iter):
        lamb, vector = projected_gradient_method(tensor_res, max_iter, tol, alpha, beta)
        lambs.append(lamb)
        vectors.append(vector)

        rank1_update = lamb * rank1_supersymmetric_tensor(vector, tensor.ndim)
        tensor_res -= rank1_update

        if np.linalg.norm(tensor_res) < tol:
            break

    return lambs, vectors

dimension = 2
order = 2
tensor = random_supersymmetric_tensor(dimension, order)

lambs, vectors = successive_rank1_decomp(tensor)

print("Original Tensor:")
print(tensor)

print("\nDecomposition Results:")
for i, (lamb, vector) in enumerate(zip(lambs, vectors)):
    print(f"Rank-1 Tensor {i+1}:")
    print(f"Lambda: {lamb}")
    print(f"Vector: {vector}")
    print()

tensor_recon = sum_rank1_supersymmetric_tensors(zip(lambs, vectors), order)

print("Reconstructed Tensor:")
print(tensor_recon)

error = np.linalg.norm(tensor - tensor_recon)
print(f"\nReconstruction Error: {error}")