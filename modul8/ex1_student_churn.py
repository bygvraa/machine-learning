import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from plotting_utils import plot_empty_values

col_names = ["Id", "Churn", "Line", "Grade", "Age", "Distance", "StudyGroup"]

file_name = 'student_churn.csv'
file_dir = os.path.dirname(__file__)
file_path = os.path.join(file_dir, file_name)

data = pd.read_csv(file_path, sep=';', header=0)


''' Data preparation '''

# x = data[ "Line" ]
# y = data[ "Age" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'k' , s = 10 )
# plt.show()

# x = data[ "Grade" ]
# y = data[ "Age" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'k' , s = 10 )
# plt.show()

# x = data[ "Line" ]
# y = data[ "Grade" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'k' , s = 10 )
# plt.show()

# x = data[ "Grade" ]
# y = data[ "Churn" ]
# plt.figure()
# plt.scatter(x.values, y.values, color = 'k' , s = 10 )
# plt.show()

# plot_empty_values(data)


''' Data cleaning '''

# Remove unused values
data.drop(['Id'], axis=1, inplace=True)

# Replace unknown values
data['StudyGroup'].fillna(data['StudyGroup'].mode()[0], inplace=True)

# Feature encoding
data['Line'] = pd.factorize(data['Line'])[0]


# Define features ('X') and target ('y')
X, y = data.drop('Churn', axis=1), data['Churn']


# Splitting into sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# Scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


''' Data model '''

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',  # 'identity', 'logistic', 'tanh', 'relu'
    max_iter=400,
    random_state=0,
    batch_size=1)


''' Data model training '''

# Train model on train sets and make predictions on test set
mlp_clf.fit(X_train, y_train)
y_pred = mlp_clf.predict(X_test)


''' Data model evaluation '''

# Evaluate model
print(classification_report(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)

print(f'Hidden layers: {mlp_clf.hidden_layer_sizes}')
print(f'Activation: {mlp_clf.activation}')
print(f'Epochs: {mlp_clf.max_iter}')

# Plot matrix
plt.matshow(matrix, cmap='Blues')

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matrix[i, j]), ha='center', va='center')

plt.yticks(range(2), ['Not Completed', 'Completed'], va='center', rotation=90)
plt.xticks(range(2), ['Not Completed', 'Completed'])

plt.title('MLP Confusion Matrix')
plt.ylabel('Actual', weight='bold')
plt.xlabel('Predicted', weight='bold')

plt.show()
