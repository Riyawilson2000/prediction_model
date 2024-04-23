import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt



# Reading dataset
df = pd.read_excel("gradient_Descent/student-mat.xlsx") #Mark Prediction
# df=pd.read_excel("gradient_Descent/diabetes_prediction_dataset_normalised.xlsx") # Diabetes Prediction


# Slicing certain columns and rows
df1 = df.loc[:,["school","age","address","studytime","failures-normalised","schoolsup","famsup","freetime-normalised","health-normalised","absences","G1","G2"]] # Mark Prediction
# df1 = df.loc[:,["gender","Age-normalised","hypertension","heart_disease","bmi-normalised","HbA1c-normalised","blood_glucose level-normalised"]]# Diabetes Prediction
# df1 = df.loc[:,["battery_power_normalised","blue","dual_sim","four_g","three_g","touch_screen","wifi"]] #Mobile Price Prediction


# Converting to numpy array(input vector)
x = df1.to_numpy()


# Creating weight vector
np.random.seed(42)
w = np.random.rand(12,1)
# w=np.clip(w,0,1)



#Slicing to get desired value
g = df.loc[:,["G3"]].to_numpy() # Mark prediction
# g = df.loc[:,["diabetes"]].to_numpy() # Diabetes Prediction
# g= df.loc[:,["price_range_normalised"]].to_numpy()# Mobile price prediction




#Looping to find desired weights

weightsGradient, fValue_gradient, iterationsGradient, weightslistGradient, functionvalues_gradient= utils.gradient_descent(x, w, g)  # 65
# fista, fValue_fista = utils.fista(x, w, g,iterations=500)                      # 26


f_gradient, _ = utils.objective_function(x, weightsGradient, g)
# f_fista, _ = utils.objective_function(x, fista, g)
print(f"The value of objective function using gradient descent : {fValue_gradient}")
print(f"Weights:{weightsGradient}")
# print(f"The value of objective function using fista : {f_fista}")

# if f_gradient > f_fista:
#     print(f"fista is better.")
# else:
#     print(f"gradient is better.")



# # gradient plot
        
# plt.plot(iterationsGradient, functionvalues_gradient, label = 'gradient descent', color ='r')
# # plt.plot(range(1, len(fValue_fista) + 1), fValue_fista, label = 'fista', color = 'b')
# plt.xlabel("Iterations")
# plt.ylabel("Objective function value")
# plt.title("Iteration vs objective function value")
# plt.legend()
# plt.show()


# # fista plot

# # plt.plot(range(1, len(fValue_fista) + 1), fValue_fista)
# # plt.xlabel("Iterations")
# # plt.ylabel("Objective function value")
# # plt.title("Loss function via FISTA")
# # plt.show()



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
