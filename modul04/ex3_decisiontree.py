import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib.colors import ListedColormap

iris = load_iris()  # load iris sample dataset
X = iris.data[:, 2:]  # petal length and width, so 2D information # type: ignore
y = iris.target  # type: ignore

# check how many samples we have
print("Number of samples: " + str(len(y)))

# visulize the dataset
plt.figure()

# define colors - red, green, blue
colormap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # type: ignore

# pplot labxel
plt.xlabel(iris.feature_names[2])  # type: ignore
plt.ylabel(iris.feature_names[3])  # type: ignore

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, edgecolor='black', s=20)

from datetime import datetime

k = str(datetime.now())
print(k)
# time the execution time of this cell.

tree_clf = DecisionTreeClassifier(max_depth=2)  # indicate we do not want the tree to be deeper than 2 levels
tree_clf.fit(X, y)  # training the classifier
print("seed : " + str(tree_clf.random_state))  # type: ignore

# prediction
print("probability of point = (2,1) = " + str(tree_clf.predict_proba([[2, 1]])))
print("probability of point = (4,1) = " + str(tree_clf.predict_proba([[4, 1]])))
print("probability of point = (5,2) =  " + str(tree_clf.predict_proba([[5, 2]])))

for name, score in zip(iris.feature_names[2:4], tree_clf.feature_importances_):  # type: ignore
    print("feature importance: ", name, score)

import pydotplus
from sklearn.externals.six import StringIO  # type: ignore
import matplotlib.image as mpimg
import io

dot_data = io.StringIO()

export_graphviz(tree_clf,
                out_file=dot_data,
                # or put a filename here filename like "graph.dot", you then need to convert it into .png
                feature_names=iris.feature_names[2:],  # type: ignore
                class_names=iris.target_names,  # type: ignore
                rounded=True,
                filled=True)

# write the dot data to a .png file
filename = "tree.png"
pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png(filename)  # type: ignore
img = mpimg.imread(filename)  # read this .png file

plt.figure(figsize=(8, 8))  # setting the size to 10 x 10 inches of the figure.
imgplot = plt.imshow(img)  # plot the image.
plt.show()

plt.figure()

# pplot labxel
plt.xlabel(iris.feature_names[2])  # type: ignore
plt.ylabel(iris.feature_names[3])  # type: ignore

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, edgecolor='black', s=20)

# drawing the splits
plt.axhline(y=1.75, color='red', linestyle='-')
plt.axvline(x=2.45, color='blue', linestyle='-')
