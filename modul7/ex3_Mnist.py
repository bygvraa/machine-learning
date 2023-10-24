import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def plot_confusion_matrix(matrix, ax=None):
    ax = ax or plt.gca()
    ax.matshow(matrix, cmap='Blues')

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')

    ax.set_yticks(range(len(matrix)))
    ax.set_xticks(range(len(matrix)))

    ax.set_ylabel('Actual', weight='bold')
    ax.set_xlabel('Predicted', weight='bold')


# Indlæs Digits-datasættet, der indeholder billeder af håndskrevne tal
mnist = load_digits()
X, y = mnist.data, mnist.target

print(f"Data Shape: {pd.DataFrame(X).shape}")
print(f'Target Shape: {pd.DataFrame(y).shape}')

# Opretter en figur med 2 rækker og 10 kolonner for at vise 20 billeder fra MNIST-datasættet.
# Hver figur viser et håndskrevet tal og dets tilsvarende target ('y').
fig, axes = plt.subplots(2, 10, figsize=(10, 4))
for i in range(20):
    # Vis billedet i gråtoner
    axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    axes[i // 10, i % 10].axis('off')
    axes[i // 10, i % 10].set_title(f"target: {y[i]}")
plt.tight_layout()
plt.show()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature scaling (standardization)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


''' Random Forest '''
rf_clf = RandomForestClassifier(
    n_estimators=23,
    max_depth=None,
    random_state=0)

rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)


''' Decision Tree '''
dt_clf = DecisionTreeClassifier(
    max_depth=10,
    random_state=0)

dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)


''' Neural Net '''
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    max_iter=400,
    random_state=0)

mlp_clf.fit(X_train, y_train)
mlp_y_pred = mlp_clf.predict(X_test)


# Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex='row')

rf_matrix = confusion_matrix(y_test, rf_y_pred)
dt_matrix = confusion_matrix(y_test, dt_y_pred)
mlp_matrix = confusion_matrix(y_test, mlp_y_pred)

plot_confusion_matrix(rf_matrix, ax=axs[0])
plot_confusion_matrix(dt_matrix, ax=axs[1])
plot_confusion_matrix(mlp_matrix, ax=axs[2])

rf_report = classification_report(y_test, rf_y_pred, output_dict=True)
dt_report = classification_report(y_test, dt_y_pred, output_dict=True)
mlp_report = classification_report(y_test, mlp_y_pred, output_dict=True)

axs[0].set_title('Random Forest Matrix', weight='bold')
axs[1].set_title('Decision Tree Matrix', weight='bold')
axs[2].set_title('MLP Confusion Matrix', weight='bold')

axs[0].text(0.5, -0.15,
            (f'Trees: {rf_clf.n_estimators}' +
             f'\nDepth: {rf_clf.max_depth}' +
             f'\nAccuracy: {rf_report["accuracy"]:.2f}'),
            ha='center',
            va='center', transform=axs[0].transAxes)
axs[1].text(0.5, -0.15,
            (f'Depth: {dt_clf.max_depth}' +
             f'\nAccuracy: {dt_report["accuracy"]:.2f}'),
            ha='center',
            va='center', transform=axs[1].transAxes)
axs[2].text(0.5, -0.15,
            (f'Layers: {mlp_clf.hidden_layer_sizes}' +
             f'\nActivation: {mlp_clf.activation}' +
             f'\nEpochs: {mlp_clf.max_iter}' +
             f'\nAccuracy: {mlp_report["accuracy"]:.2f}'),
            ha='center',
            va='center', transform=axs[2].transAxes)

plt.tight_layout()
plt.show()
