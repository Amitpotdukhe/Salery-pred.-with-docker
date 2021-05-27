import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
x = dataset['YearsExperience'].values.reshape(-1,1)
y = dataset['Salary']

mind = LinearRegression()

model = mind.fit(x,y)

print('----Welcome to Salery Predictor App----')
a = input('Enter number of Years:')
b = float(a)

p = model.predict([[b]])
print("Predicted Salery is:", p)
