### DIABETES PREDICTION ###

## Dataset: diabetes_prediction_dataset_normalised.xlsx


# Importing libraries
import numpy as np
import pandas as pd
from utils import gradient_descent, fista, objective_function


# Load the dataset
df = pd.read_excel('gradient_Descent/diabetes_prediction_dataset_normalised.xlsx')


# X
x = df.loc[1 : 50000,["gender","Age-normalised","hypertension","heart_disease",
              "bmi-normalised","HbA1c-normalised",
              "blood_glucose level-normalised"]].to_numpy(dtype=np.float16)


# G
g = df.loc[:,["diabetes"]].to_numpy(dtype=np.int8)

# W
w = np.zeros((7,), dtype=np.int8)

gradient, fValue_gradient, _ = gradient_descent(x, w, g, 100)
print("OKAY")
