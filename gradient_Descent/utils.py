# importing necessary libraries
# importing necessary libraries
import numpy as np


# Defining Objective function
def objective_function(x, w, g):
    z = np.dot(x, w)
    # y = 1/(1 + np.exp(-z))

    return np.mean(np.square(z - g)), z


def gradient_descent_while(x, w, g, learning_step=1 / 3, epsilon=1e-3):
    # Initialising
    h = learning_step
    f_value = []
    weights = []
    previous_f = None
    iterations = 0

    # Estimating optimal parameters
    while True:
        funct, z = objective_function(x, w, g)
        n = x.shape[0]

        if (
            previous_f
            and (previous_f - funct) < 0
            and abs(previous_f - funct) <= epsilon
        ):
            break

        previous_f = funct

        # if abs(previous_f) <= epsilon:
        #     break

        f_value.append(funct)
        weights.append(w)

        # gradient =  -(2/ n) * np.sum(x * (g - y))
        gradient = (2 / n) * np.dot(x.T, (z - g))

        # Updating weights
        w = w - (h * gradient)

        iterations += 1

    print(f"Iteration number: {iterations}")
    return w, f_value, weights


# Defining gradient descent function
def gradient_descent(x, w, g, iterations=100, learning_step=1 / 3, epsilon=1e-3):
    # Initialising
    iterations = iterations
    h = learning_step
    iterations = iterations
    h = learning_step
    f_value = []
    weights = []
    previous_f = None

    # Estimating optimal parameters
    # Estimating optimal parameters
    for i in range(iterations):
        funct, z = objective_function(x, w, g)
        funct, z = objective_function(x, w, g)
        n = x.shape[0]

        # if previous_f and (previous_f - funct) < 0:
        #     break

        previous_f = funct

        if abs(previous_f) <= epsilon:
            break

        f_value.append(funct)
        weights.append(w)

        # gradient =  -(2/ n) * np.sum(x * (g - y))
        gradient = (2 / n) * np.dot(x.T, (z - g))

        # Updating weights
        w = w - (h * gradient)

    return w, f_value, weights


# Define the FISTA function
# def fista(x, w, g, L=3, iterations=100):

#     w_0 = w          # w_k-1
#     y_1 = w_0
#     t_k = 1          # t_k+1
#     w_k = w_0        # w_k
#     t_1 = 1          # t_k
#     h = 1 / L

#     function_list = []

#     for k in range(iterations):
#         gradient_w_k = gradient(x, w_0, g)
#         w_0, w_k = w_k, y_1- h * gradient_w_k  
#         t_1, t_k = t_k, 1 + np.sqrt((1 + 4 * (t_1)**2)) / 2
#         y_1 = w_k + ((t_1 -1) / t_k) * (w_k - w_0)
#         w_k = np.clip(w_k,0,1)
#         print(w_k)
#         print(f"Iteration {k+1}: w_{k+1} {w_k}")
#         funct, _ = objective_function(x, w_k, g)

#         function_list.append(funct)

#         z = np.dot(x, w_k)
#         y = 1/(1 + np.exp(-z))
#         # n = y.shape[0]
#         # g = g.reshape((n, 1))

#         print(f"value of f_fista at {k+1} iteration: {np.mean(np.square(y - g))}")
        
#     return w_k, function_list


if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    w = np.ones((3, 1))
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    w = np.ones((3, 1))
    g = np.array([0.1, 0.2, 0.5])

    weight, f_value, weights = gradient_descent(x, w, g, learning_step=0.1)
