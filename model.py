import pandas
import numpy
from sklearn.linear_model import LinearRegression

ds=pandas.read_csv('Salary.csv')
x=ds['YearExperience'].values.reshape(35,1)
y=ds['Salary']

model=LinearRegression
model.fit(x,y)

z=float(input("Please Enter the experience through which predicted salary will be displayed:"))
result=model.predict( [[z]] )

print(result)
