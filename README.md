# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages (numpy, pandas, SVC, etc).
2. Load the dataset and vectorize text information using CountVectorizer()
3. Fit the data into the model.
4. Check Accuracy Score and Confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shri Sai Aravind. R
RegisterNumber: 212223040197  
*/
```
```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

print(f"{df.info()} \n\n{df.head()} \n\n{df.isnull().sum()}\n")

x = df["v2"].values
y = df["v1"].values

x_train, x_test, y_train, y_test =  train_test_split(x,y,random_state=0)
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

model = SVC(kernel='linear', random_state=0).fit(x_train, y_train)

y_pred = model.predict(x_test)
res_df = pd.DataFrame({"Predicted": y_pred, "Actual": y_test})

print(f"\n{res_df}\n")
print(f"Accuracy Score is {accuracy_score(y_test, y_pred)*100:.2f}\n")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

## Output:
![image](https://github.com/user-attachments/assets/72231249-604e-4e5d-843f-e415f0005927)

![image](https://github.com/user-attachments/assets/46e6b111-fd86-4b77-bbfb-051673ebd37a)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
