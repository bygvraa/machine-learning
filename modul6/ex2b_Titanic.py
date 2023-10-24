# Titanic dataset predictions

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


# Construct the full file path from the current directory
file_name = 'titanic_800.csv'
file_dir = os.path.dirname(__file__)
file_path = os.path.join(file_dir, file_name)

# Read the .csv file, while skipping the header
data = pd.read_csv(file_path, sep=',', header=0)

# Remove unused values
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Replace unknown values
data['Survived'].fillna(data['Survived'].median().astype(int), inplace=True)
data['Pclass'].fillna(data['Pclass'].median().astype(int), inplace=True)
data['Age'].fillna(data['Age'].median().astype(int), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Feature encoding
data['Sex'].replace({'female': 0, 'male': 1}, inplace=True)
data['Embarked'] = pd.factorize(data['Embarked'])[0]

# Define features ('X') and target ('y')
X, y = data.drop('Survived', axis=1), data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature scaling (standardization)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create a classifier
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',  # 'identity', 'logistic', 'tanh', 'relu'
    max_iter=400,
    random_state=0,
    batch_size=1)

# Train model and make predictions on test set
mlp_clf.fit(X_train, y_train.values.ravel())
y_pred = mlp_clf.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
mlp_params = mlp_clf.get_params()

print(f'Hidden layers: {mlp_params["hidden_layer_sizes"]}')
print(f'Activation: {mlp_params["activation"]}')
print(f'Epochs: {mlp_params["max_iter"]}')


# Plot matrix
plt.matshow(matrix, cmap='Blues')

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matrix[i, j]), ha='center', va='center')

plt.yticks(range(2), ['Not Survived', 'Survived'], va='center', rotation=90)
plt.xticks(range(2), ['Not Survived', 'Survived'])

plt.title('MLP Confusion Matrix')
plt.ylabel('Actual', weight='bold')
plt.xlabel('Predicted', weight='bold')

plt.show()
