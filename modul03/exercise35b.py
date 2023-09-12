from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris["data"][:, (2,3)]  # petal length, petal width # type: ignore
yy = iris["target"] # type: ignore


y = []

for i in yy: 
  if i == 2:
     y = np.append(y,[1])
  else:
     y = np.append(y,[0])


#model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
model = LogisticRegression(C=1000)
model.fit(X,y)


plt.plot(X[:, 0][y==1], X[:, 1][y==1], "y.", label="Virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.", label="Setosa or versicolor")

plt.legend(loc="upper left", fontsize=14)

plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)


xx = np.linspace(1, 6)
yy = -  (model.coef_[0,0] / model.coef_[0,1]) * xx - ( model.intercept_[0]  /  model.coef_[0,1])


plt.plot(xx, yy, 'k-')

plt.show()