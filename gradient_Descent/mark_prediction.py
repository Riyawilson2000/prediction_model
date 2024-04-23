## STUDENT MARK PREDICTION ##

## Dataset: student-mat.xlsx


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import gradient_descent_while, fista, objective_function


# Load the dataset
df = pd.read_excel("gradient_Descent/student-mat1.xlsx")


# X
x = df.loc[
    :, ["G1", "G2", "age", "absences", "studytime", "address", "school","failures-normalised","schoolsup","famsup","freetime-normalised","health-normalised"]].to_numpy()

#
# G
g = df.loc[:, ["G3"]].to_numpy()

# W
w = np.zeros((12, 1))


# gradient descent and fista algorithms - comparison
gradient, fValue_gradient, weightList_gradient = gradient_descent_while(
    x, w, g, epsilon=1e-10
)
# fista, fValue_fista = fista(x, w, g, iterations=100)


f_gradient, _ = objective_function(x, gradient, g)


print(f"Gradient Descent objective function value: {f_gradient}")
print(f"Gradient Descent weights: {gradient}")

plt.plot(
    range(1, len(fValue_gradient) + 1),
    fValue_gradient,
    label="gradient descent",
    color="r",
)
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.title("Iteration vs objective function value")
plt.legend()
plt.show()