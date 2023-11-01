import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from plotting_utils import plot_confusion_matrix, plot_empty_values

file_name = 'student_churn.csv'
file_dir = os.path.dirname(__file__)
file_path = os.path.join(file_dir, file_name)

data = pd.read_csv(file_path, sep=';', header=0)


''' Data preparation '''

# Loop through the subplots and plot the data
fig, axs = plt.subplots(3, 2)
features = ['Line', 'Grade', 'Age', 'Distance', 'StudyGroup']
for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    ax = axs[row, col]
    for churn_type in ['Completed', 'Stopped']:
        ax.hist(data[data['Churn'] == churn_type]
                [feature], alpha=0.5, label=churn_type)
    ax.set_xlabel(feature)
    ax.legend()
plt.tight_layout()
plt.show()

plot_empty_values(data)


''' Data cleaning '''

# Remove unused values
data.drop(['Id'], axis=1, inplace=True)

# Replace unknown values
data['StudyGroup'].fillna(data['StudyGroup'].mode()[0], inplace=True)
data['Line'] = pd.factorize(data['Line'])[0]


''' Data preprocessing '''

# Define features ('X') and target ('y')
X, y = data.drop('Churn', axis=1), data['Churn']

# Split into sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


''' Model setup '''

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',  # 'identity', 'logistic', 'tanh', 'relu'
    max_iter=400,
    random_state=0,
    batch_size=1)


''' Model training '''

# Train model on train sets and make predictions on test set
mlp_clf.fit(X_train, y_train)
mlp_y_pred = mlp_clf.predict(X_test)


''' Model evaluation '''

# Evaluate model
print(classification_report(y_test, mlp_y_pred))
mlp_matrix = confusion_matrix(y_test, mlp_y_pred)
tn, fp, fn, tp = mlp_matrix.ravel()

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f'Hidden layers: {mlp_clf.hidden_layer_sizes}')
print(f'Activation: {mlp_clf.activation}')
print(f'Epochs: {mlp_clf.max_iter}')

# Plot matrix
plot_confusion_matrix(mlp_matrix, labels=['Not Completed', 'Completed'])

plt.title('MLP Confusion Matrix')
plt.show()
