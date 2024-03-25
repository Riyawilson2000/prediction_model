## STUDENT MARK PREDICTION ##

## Dataset: student-mat.xlsx


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import gradient_descent, fista, objective_function


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


## Gradient plot
plt.plot(
    range(1, len(fValue_gradient) + 1),
    fValue_gradient,
    label="gradient descent",
    color="r",
)
plt.plot(range(1, len(fValue_fista) + 1), fValue_fista, label="fista", color="b")
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.title("Iteration vs objective function value")
plt.legend()
plt.show()


## fista plot
plt.plot(range(1, len(fValue_fista) + 1), fValue_fista)
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.title("Loss function via FISTA")
plt.show()
