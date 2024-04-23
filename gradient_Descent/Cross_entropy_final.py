import pandas as pd
import numpy as np
import Cross_entropy
import matplotlib.pyplot as plt



# Reading dataset
# df = pd.read_excel("gradient_Descent/student-mat.xlsx") #Mark Prediction
df=pd.read_excel("gradient_Descent/diabetes_prediction_dataset_normalised.xlsx") # Diabetes Prediction


# Slicing certain columns and rows
# df1 = df.loc[:,["G1", "G2", "age", "absences","studytime", "address", "school"]] # Mark Prediction
df1 = df.loc[:,["gender","Age-normalised","hypertension","heart_disease","bmi-normalised","HbA1c-normalised","blood_glucose level-normalised","smoking_history"]]# Diabetes Prediction


# Converting to numpy array(input vector)
x = df1.to_numpy()
print(x)

# Creating weight vector
np.random.seed(42)
w = np.random.rand(8,1)
# w=np.clip(w,0,1)
# w=1/(1+(np.exp(-w)))
# w = np.maximum(0, np.minimum(1, w))



#Slicing to get desired value
# g = df.loc[:,["G3"]].to_numpy() # Mark prediction
g = df.loc[:,["diabetes"]].to_numpy() # Diabetes Prediction


print(x.shape)
print(w.shape)
print(g.shape)
print(g)


#Looping to find desired weights

gradient, fValue_gradient, _ = Cross_entropy.gradient_descent(x, w, g,100) 
                   


f_gradient, _ = Cross_entropy.binary_cross_entropy(x, gradient, g)

print(f"The value of objective function using gradient descent : {f_gradient}")





# # gradient plot
        
plt.plot(range(1, len(fValue_gradient) + 1), fValue_gradient, label = 'gradient descent', color ='r')
# plt.plot(range(1, len(fValue_fista) + 1), fValue_fista, label = 'fista', color = 'b')
plt.xlabel("Iterations")
plt.ylabel("Objective function value")
plt.title("Iteration vs objective function value")
plt.legend()
plt.show()


