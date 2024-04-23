## STUDENT MARK PREDICTION ##

## Dataset: student-mat.xlsx


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import gradient_descent_while, fista, objective_function


# Load the dataset
df = pd.read_excel("gradient_Descent/student-mat.xlsx")


# X
x = df.loc[
    :, ["G1", "G2", "age", "absences", "studytime", "address", "school"]
].to_numpy()


# G
g = df.loc[:, ["G3"]].to_numpy()

# W
w = np.zeros((7, 1))


# gradient descent and fista algorithms - comparison
gradient, fValue_gradient, weightList_gradient = gradient_descent_while(
    x, w, g, epsilon=1e-1
)
# fista, fValue_fista = fista(x, w, g, iterations=100)


f_gradient, _ = objective_function(x, gradient, g)


print(f"Gradient Descent objective function value: {f_gradient}")
print(f"Gradient Descent weights: {gradient}")
