import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from plotting_utils import plot_confusion_matrix

raw_data = datasets.load_wine()

print(raw_data.DESCR)

# There are apparently 3 classes (creatively named 'class_0', 'class_1', and 'class_2').
# Probably... these correspond to some typical wine varietals like Pinot Noir, or Cabernet, or Merlot...

# As this is a dictionary, we will print out the key/value pairs
# so we can decide how we'll format a data structure useful for our needs
# for key, value in raw_data.items():
#     print(key, '\n', value, '\n')


# We have 178 samples (rows) and 13 features (columns).
print('data.shape\t', raw_data['data'].shape,
      '\ntarget.shape \t', raw_data['target'].shape)


''' Data preparation '''

target_names = raw_data.target_names

X, y = raw_data.data, raw_data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


''' Model setup '''


mlp_clf = MLPClassifier(
    hidden_layer_sizes=(6,),
    activation='relu',
    solver='lbfgs',
    alpha=1e-05,
    max_iter=10000,
    random_state=1)


''' Model training '''

mlp_clf.fit(X_train, y_train)
y_pred_train = mlp_clf.predict(X_train)
y_pred_test = mlp_clf.predict(X_test)


''' Model evaluation '''

train_score = accuracy_score(y_train, y_pred_train)
print("score on train data: ", train_score)

test_score = accuracy_score(y_test, y_pred_test)
print("score on test data: ", test_score)

print(classification_report(y_test, y_pred_test, target_names=target_names))
mlp_matrix = confusion_matrix(y_test, y_pred_test)

print(f'Hidden layers: {mlp_clf.hidden_layer_sizes}')
print(f'Activation: {mlp_clf.activation}')
print(f'Epochs: {mlp_clf.max_iter}')

# Plot matrix
plot_confusion_matrix(mlp_matrix, labels=target_names)

plt.title('MLP Confusion Matrix')
plt.show()
