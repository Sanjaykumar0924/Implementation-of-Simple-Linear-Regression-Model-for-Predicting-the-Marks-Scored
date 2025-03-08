# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJAY KUMAR H
RegisterNumber:  212223040182
```










```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```


## Output:
![image](https://github.com/user-attachments/assets/c0fe7649-039c-44d9-989a-2c6b1e7dd0d4)
![image](https://github.com/user-attachments/assets/7de0be1b-c8ce-4555-9b8a-e1dfcf47174e)
![image](https://github.com/user-attachments/assets/7896a547-4222-4cf9-8267-574447b2b3db)
![image](https://github.com/user-attachments/assets/1281b970-e708-4cab-8123-e45ed7666f52)
![image](https://github.com/user-attachments/assets/447d1749-7a59-4429-ac18-cc4084f65660)
![image](https://github.com/user-attachments/assets/948c9a9d-66be-4847-af02-f3b4096d0ad0)

![image](https://github.com/user-attachments/assets/00da63e9-78e7-43ea-874b-3dd7fc27a89e)

![image](https://github.com/user-attachments/assets/27fae159-4d9f-4a61-96eb-a6c4e5d0ee1f)
![image](https://github.com/user-attachments/assets/bab9d0b6-6f4c-42f0-bcb1-1eca5e1206bc)
![image](https://github.com/user-attachments/assets/720f2358-2bd1-402e-a57a-dbd112c4865c)
![image](https://github.com/user-attachments/assets/60faa2b8-7571-4207-9a6a-d5bfd5ce1461)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
