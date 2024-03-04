### DIABETES PREDICTION ###

## Dataset: diabetes_prediction_dataset_normalised.xlsx


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import gradient_descent, fista, objective_function


# Load the dataset
df = pd.read_excel('gradient_Descent/diabetes_prediction_dataset_normalised.xlsx')


# X
x = df.loc[1 : 500,["gender","Age-normalised","hypertension","heart_disease",
              "bmi-normalised","HbA1c-normalised",
              "blood_glucose level-normalised"]].to_numpy(dtype=np.float16) # 500 rows


# G
g = df.loc[1 : 500,["diabetes"]].to_numpy(dtype=np.int8)

# W
w = np.zeros((7,1), dtype=np.int8)


# gradient descent and fista algorithms - comparison
gradient, fValue_gradient, _ = gradient_descent(x, w, g, 100)
fista, fValue_fista = fista(x, w, g, iterations=100)


f_gradient, _ = objective_function(x, gradient, g)
f_fista, _ = objective_function(x, fista, g)
print(f"The value of objective function using gradient descent : {f_gradient}")
print(f"The value of objective function using fista : {f_fista}")

if f_gradient > f_fista:
    print("fista is better.")
else:
    print("gradient is better.")


## Objective function value plot     
plt.plot(range(1, len(fValue_gradient) + 1), 
         fValue_gradient, label = 'gradient descent', color ='r')
plt.plot(range(1, len(fValue_fista) + 1), fValue_fista, label = 'fista', color = 'b')
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.title("Iteration vs objective function value")
plt.legend()
plt.show()