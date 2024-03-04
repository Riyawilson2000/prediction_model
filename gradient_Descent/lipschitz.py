import numpy as np

def compute_lipschitz_constant(x):
    n = x.shape[0]
    np.random.seed(10)
    w = np.random.rand(x.shape[1]) # Random initialization of weights for computing finite differences
    epsilon = 1e-6  # Small value for finite differences

    # Compute the gradient at the initial point
    z = np.dot(x, w)
    y = 1/(1 + np.exp(-z))
    # loss_initial, _ = np.mean(np.square(y - g)), y
    gradient_initial = np.dot(x.T, 2 * (y - g) * y * (1 - y)) / n

    # Perturb the weights and compute the gradient at the perturbed point
    w_perturbed = w + epsilon
    z_perturbed = np.dot(x, w_perturbed)
    y_perturbed = 1/(1 + np.exp(-z_perturbed))
    # loss_perturbed, _ = np.mean(np.square(y_perturbed - g)), y_perturbed
    gradient_perturbed = np.dot(x.T, 2 * (y_perturbed - g) * y_perturbed * (1 - y_perturbed)) / n

    # Compute the Lipschitz constant
    lipschitz_constant = np.max(np.abs(gradient_perturbed - gradient_initial) / epsilon)

    return lipschitz_constant

# Example usage
np.random.seed(100)
x = np.random.rand(10, 5) 
np.random.seed(100) # Replace with your actual input data
g = np.random.rand(10)  # Replace with your actual target values
lipschitz_constant = compute_lipschitz_constant(x)
print("Lipschitz constant:", lipschitz_constant)
