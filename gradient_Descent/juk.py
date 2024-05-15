import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Cross_entropy
import utils

# def gradient_descent_while(x, w, g, epsilon=1e-2, learning_step=1/3):
#     f_values = []
#     weights = []
#     previous_f = float('inf')
#     iterations = 0


#     while True:
#         funct, z = Cross_entropy.binary_cross_entropy(x, w, g)
        

#         if abs(previous_f - funct) <= epsilon:
#             break
        
        
#         previous_f = funct 

        
#         f_values.append(funct)
#         weights.append(w)

#         gradient = np.dot(x.T, (z - g))/ len(g)
#         w = w-( learning_step * gradient)

#         iterations +=1

        

#     print(f"Iteration number: {iterations}")
#     return w, f_values, weights

# Reading dataset
df = pd.read_excel("gradient_Descent/diabetes_prediction_dataset_normalised1.xlsx")

# Slicing certain columns and rows
df1 = df.loc[:, ["gender", "Age-normalised", "hypertension", "heart_disease", "bmi-normalised", "HbA1c-normalised", "blood_glucose level-normalised", "smoking_history"]]

# Converting to numpy array(input vector)
x = df1.to_numpy()

# Creating weight vector
w = np.zeros((8, 1))

# Slicing to get desired value
g = df.loc[:, ["diabetes"]].to_numpy()

# Gradient descent
gradient, f_value_gradient, _ = utils.gradient_descent_while(x, w, g, epsilon=1e-4)

# Calculate objective function value
f_gradient, _ = Cross_entropy.binary_cross_entropy(x, gradient, g)

print(f"The value of objective function using gradient descent: {f_gradient}")
print(f"Gradient Descent weights: {gradient}")

# # Plotting
# plt.plot(range(1, len(f_value_gradient) + 1), f_value_gradient, label='Gradient Descent', color='r')
# plt.xlabel("Iterations")
# plt.ylabel("Objective function value")
# plt.title("Iteration vs Objective function value")
# plt.legend()
# plt.show()
