# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm

1. Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: P.SYAM TEJ
RegisterNumber:212221240056  
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

![8 1](https://user-images.githubusercontent.com/93427224/173561703-b8bbae8b-16a8-422a-ab01-623efd4a46fc.png)

![8 2](https://user-images.githubusercontent.com/93427224/173561763-7995d2a8-81c7-4e8d-857e-e7bbe6860141.png)

![8 3](https://user-images.githubusercontent.com/93427224/173561787-45015f30-c2ab-46c0-9214-9f7875affe04.png)

![8 4](https://user-images.githubusercontent.com/93427224/173561806-3c0a946f-68f9-46b8-a4c4-32138ccb5828.png)

![8 5](https://user-images.githubusercontent.com/93427224/173561834-6339eb0b-c352-431f-ad66-ddcf76fe03cc.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
