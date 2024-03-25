import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt


# Reading dataset
df = pd.read_excel("gradient_Descent/student-mat.xlsx")  # Mark Prediction
# df=pd.read_excel("gradient_Descent/diabetes_prediction_dataset_normalised.xlsx") # Diabetes Prediction
# df=pd.read_csv("gradient_Descent/Mobile Price prediction dataset.csv") #Mobile Price Prediction

# Slicing certain columns and rows
df1 = df.loc[
    :, ["G1", "G2", "age", "absences", "studytime", "address", "school"]
]  # Mark Prediction
# df1 = df.loc[:,["gender","Age-normalised","hypertension","heart_disease","bmi-normalised","HbA1c-normalised","blood_glucose level-normalised"]]# Diabetes Prediction
# df1 = df.loc[:,["battery_power_normalised","blue","dual_sim","four_g","three_g","touch_screen","wifi"]] #Mobile Price Prediction


# Converting to numpy array(input vector)
x = df1.to_numpy()


# Creating weight vector
np.random.seed(42)
w = np.random.rand(7, 1)
# print(w)
# w = np.zeros((7,1))


# Slicing to get desired value
g = df.loc[:, ["G3"]].to_numpy()  # Mark prediction
# g = df.loc[:,["diabetes"]].to_numpy() # Diabetes Prediction
# g= df.loc[:,["price_range_normalised"]].to_numpy()# Mobile price prediction

# print(x.shape)
# print(w.shape)
# print(g.shape)
# print(g)


# Looping to find desired weights

gradient, fValue_gradient, _ = utils.gradient_descent(x, w, g, 100)  # 65
# fista, fValue_fista = utils.fista(x, w, g, iterations=100)  # 26


f_gradient, _ = utils.objective_function(x, gradient, g)
# f_fista, _ = utils.objective_function(x, fista, g)
print(f"The value of objective function using gradient descent : {f_gradient}")
# print(f"The value of objective function using fista : {f_fista}")

# if f_gradient > f_fista:
#     print(f"fista is better.")
# else:
#     print(f"gradient is better.")


# # gradient plot

# plt.plot(
#     range(1, len(fValue_gradient) + 1),
#     fValue_gradient,
#     label="gradient descent",
#     color="r",
# )
# plt.plot(range(1, len(fValue_fista) + 1), fValue_fista, label="fista", color="b")
# plt.xlabel("Iterations")
# plt.ylabel("Objective function value")
# plt.title("Iteration vs objective function value")
# plt.legend()
# plt.show()


# fista plot

# plt.plot(range(1, len(fValue_fista) + 1), fValue_fista)
# plt.xlabel("Iterations")
# plt.ylabel("Objective function value")
# plt.title("Loss function via FISTA")
# plt.show()


# z = np.dot(x,w)
# y = 1/(1 + np.exp(-z))

# # Plot zi vs gi graph
# plt.scatter(y , g, label='zi vs gi')
# plt.xlabel("zi values")
# plt.ylabel("gi values")
# plt.title("zi vs gi graph")
# plt.legend()
# plt.show()

# #Plot xi vs Gi graph
# plt.scatter(range(1, len(g) + 1), g, color='orange', label='xi vs Gi')
# plt.xlabel("xi")
# plt.ylabel("Gi values")
# plt.title("xi vs Gi graph")
# plt.legend()
# plt.show()


# #  0.31999990253440114  --- 227  --  0.0035
