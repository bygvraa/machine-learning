import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Indlæs Digits-datasettet, som indeholder billeder af håndskrevne tal
mnist = load_digits()

print("Data Shape:")  # Udskriv antal rækker og kolonner af datasættet
print(pd.DataFrame(mnist.data).shape)

print("Data Head:")
print(pd.DataFrame(mnist.data).head())

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

# Opretter en Random Forest Classifier-model med 3 træer
# og ingen maksimal dybde (dybde er ikke begrænset).
clf = RandomForestClassifier(n_estimators=3, max_depth=None)

clf.fit(X_train, y_train)

# Laver forudsigelser på testdataene
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

mat = confusion_matrix(y_test, y_pred)
print(mat)

# Opret et heatmap med seaborn for at visualisere confusion matrix
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
