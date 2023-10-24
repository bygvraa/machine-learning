# Titanic dataset predictions

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


def accuracy(tp, tn, fp, fn):
    correct_predictions = tp + tn
    all_predictions = tp + tn + fp + fn
    accuracy = correct_predictions / all_predictions
    return accuracy


# Get the directory of the currently executing script
script_dir = os.path.dirname(__file__)

# Define the file name
file_name = 'titanic_train_500_age_passengerclass.csv'

# Construct the full file path
file_path = os.path.join(script_dir, file_name)

# Read the .csv file, while skipping the header
data = pd.read_csv(file_path, sep=',', header=0)

# show the data
print(data.describe(include='all'))

# the 'describe' is a great way to get an overview of the data
# print(data.values)

# Replace unknown values. Unknown class set to 3
data["Pclass"].fillna(3, inplace=True)

# Replace unknown values. Unknown age set to 25
data["Age"].fillna(25, inplace=True)

# Replace unknown values. Unknown survival set to survived
data["Survived"].fillna(1, inplace=True)


yvalues = pd.DataFrame(dict(Survived=[]), dtype=int)
yvalues["Survived"] = data["Survived"].copy()
# now the yvalues should contain just the survived column

x = data["Age"]
y = data["Pclass"]

plt.figure()
plt.scatter(x.values, y.values, color='black', s=20)
# plt.show()

# now we can delete the survived column from the data (because
# we have copied that already into the yvalues).
data.drop('Survived', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)

# Display updated dataset
print(data.describe(include='all'))

# Split the dataset into training and testing sets
xtrain = data.head(400)
xtest = data.tail(100)

ytrain = yvalues.head(400)
ytest = yvalues.tail(100)


# Feature scaling (standardization)
scaler = StandardScaler()
scaler.fit(xtrain)

xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)


# Create an MLPClassifier (Multi-Layer Perceptron) with parameters
hidden_layer_sizes = (8, 8)  # 2 hidden layers, each with 8 neurons
activation = "identity"  # 'identity', 'logistic', 'tanh', 'relu'
epochs = 100

mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    max_iter=epochs,
    random_state=0,
    batch_size=1)

# Train the MLPClassifier model
mlp.fit(xtrain, ytrain.values.ravel())
# the reason for the values.ravel is that these data has not been scaled, and they need to be converted to the
# correct input format for the mlp.fit. Data that is scaled already has this done to them.

# Make predictions on the test set ('xtest')
predictions = mlp.predict(xtest)

# Evaluate the model's performance using confusion matrix and classification report
matrix = confusion_matrix(ytest, predictions)

# print(matrix)
print(classification_report(ytest, predictions))


# Extract true positives, true negatives, false positives, and false negatives
tn, fp, fn, tp = matrix.ravel()

print("TP: %f" % tp)
print("TN: %f" % tn)
print("FP: %f" % fp)
print("FN: %f" % fn)
print("Accuracy: %f" % accuracy(tp, tn, fp, fn))
print(f"Hidden layers: {hidden_layer_sizes}")
print(f"Activation: {activation}")
print(f"Epochs: {epochs}")

'''
Experiments below - be systematic and record your result of each experiment with both the value of all important 
parameters used and the achieved accuracy. Now it is time for some experiments (remember to record, write down, 
your results - each time.

Be systematic in your experiments):
Try (at least) the following (in a systematic way, record your findings):
  a) Try to experiment with the network topology (number of nodes and layers) to see if you can get a higher accuracy.
      - 0.73 (8, 8)

  b) Try to experiment with the way you handle the missing input age data to see how this reflect the accuracy?

  c) Can we do without the scaler?
      - Yes, but the result is not as accurate

  d) What about the batch size? Experiment with that also and see if it has an effect on the accuracy.

  e) What about the number of epochs? Does that have an effect on accuracy?
      - 100 seems to be a good iteration

  f) What about changing the activation function? (again see the MLP scikit documentation for that - link earlier)
      - 0.73: relu
      - 0.74: tanh
      - 0.76: logistic
      - 0.77: identity
'''
