
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split

diabetes = datasets.load_diabetes()
x = diabetes.data[:,0];
x = x.reshape((x.shape[0],1));
y = diabetes.target
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0)
lin_reg = linear_model.LinearRegression()
lin_reg.fit(xtrain, ytrain);