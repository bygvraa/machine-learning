import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Indlæs Digits-datasettet, som indeholder billeder af håndskrevne tal
mnist = load_digits()

print("Data Shape:")  # Udskriv antal rækker og kolonner af datasættet
print(pd.DataFrame(mnist.data).shape)

print("Data Head:")
# print(pd.DataFrame(mnist.data).head())

print("Target Shape:")
print(pd.DataFrame(mnist.target).shape)

# Opretter en figur med 2 rækker og 10 kolonner for at vise 20 billeder fra MNIST-datasettet.
# Hver figur viser et håndskrevet tal og dets tilsvarende måletiket.
    # fig, axes = plt.subplots(2, 10, figsize=(16, 6))
    # for i in range(20):
    #     # Vis billedet i gråtoner
    #     axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    #     axes[i // 10, i % 10].axis('off')
    #     axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")

    # plt.tight_layout()
    # plt.show()

X_train, X_test, y_train, y_test = train_test_split(mnist.data,
                                                    mnist.target,
                                                    test_size=0.2,
                                                    random_state=0)

# Set the size of the plot
plt.figure(figsize=(14, 7))

### Random Forest
# Opretter en Random Forest Classifier-model med 3 træer
# og ingen maksimal dybde (dybde er ikke begrænset).
n_trees = 3
depth = None

rf_clf = RandomForestClassifier(
    n_estimators=n_trees,
    max_depth=depth)

rf_clf.fit(X_train, y_train)

rf_y_pred = rf_clf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_y_pred)
rf_cm = confusion_matrix(y_test, rf_y_pred)

# Opret et heatmap med seaborn for at visualisere confusion matrix
plt.subplot(1, 2, 1)
sns.heatmap(rf_cm,
            fmt='d',
            annot=True)

plt.title('Random Forest' +
          f'\nTrees: {n_trees}, depth: {depth}' +
          f'\nRF Accuracy: {rf_acc}')
plt.ylabel('Actual')
plt.xlabel('Predicted')


### MLP
n_layers = (8, 8)
n_epochs = 100

mlp_clf = MLPClassifier(
    hidden_layer_sizes=n_layers, 
    max_iter=n_epochs)

mlp_clf.fit(X_train, y_train)
mlp_y_pred = mlp_clf.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_y_pred)
mlp_cm = confusion_matrix(y_test, mlp_y_pred)

plt.subplot(1, 2, 2)
sns.heatmap(mlp_cm,
            fmt='d',
            annot=True)

plt.title('Multilayer Perceptron (MLP)' +
          f'\nLayers: {n_layers}, epochs: {n_epochs}' +
          f'\nMLP Accuracy: {mlp_acc}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.show()
