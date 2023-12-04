import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
Run the program and consider the following questions:
1. Look at the data points on the graph. Do age and blood pressure appear to have a linear relationship?
2. What does the value of r tell you about the relationship between age and blood pressure?
'''

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values


x = x.reshape(-1, 1)

model = LinearRegression().fit(x, y)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
x_predict = 43

prediction = model.predict([[x_predict]])

print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")


#sets the size of the graph
plt.figure(figsize=(5,4))

plt.scatter(x,y, c="purple")
plt.scatter(x_predict, prediction, c="blue")

#labels axes and creates scatterplot
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x, y)

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")


plt.legend()
plt.show()
