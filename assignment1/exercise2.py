import math
import matplotlib.pyplot as plt


def plot_survival_distribution(data, features):
    colors = ['lightcoral', 'royalblue']
    num_features = len(features)

    num_rows = math.ceil(num_features / 2)
    num_cols = 2 if num_features > 1 else 1

    fig, axs = plt.subplots(num_rows, num_cols, sharey='row')
    axs = axs.flatten() if num_features > 1 else axs

    for i, feature in enumerate(features):
        ax = axs[i] if num_features > 1 else axs

        # Calculate total passengers' distribution
        total_feature = data[feature].value_counts().sort_index()
        ax.bar(total_feature.index, total_feature.values,
               color=colors[1], alpha=0.6, label='Total Passengers')

        # Calculate not survived passengers' distribution
        not_survived_subset = data[data['Survived'] == 0]
        not_survived_feature = not_survived_subset[feature].value_counts(
        ).sort_index()
        ax.bar(not_survived_feature.index, not_survived_feature.values,
               color=colors[0], alpha=0.6, label='Not Survived')

        # Customize plot labels
        ax.set_title(f'Survival Distribution by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')

    # Remove any unused subplots
    for i in range(num_features, num_rows * num_cols):
        fig.delaxes(axs[i])

    # Adjust layout and display the main plot
    plt.tight_layout()
    plt.show()


def plot_empty_values(data):
    # Calculate the count of empty values for each feature
    empty_values = data.isnull().sum()

    # Create a horizontal bar plot
    plt.figure(figsize=(6, 4))
    empty_values.plot(kind='barh', color='royalblue')

    # Set the x-axis limit to the maximum number of data points
    plt.xlim(0, len(data))

    plt.title('Number of Empty Values in Dataset')
    plt.xlabel('Count')
    plt.ylabel('Feature')

    # Annotate the bars with the count of empty values
    for feature, value in enumerate(empty_values):
        plt.text(value, feature, str(value), va='center')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix, name='Plot'):
    plt.figure(figsize=(4, 4))
    plt.matshow(matrix, cmap='Blues', fignum=1)

    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')

    classes = ["Not Survived", "Survived"]
    tick_marks = range(len(classes))

    plt.yticks(tick_marks, classes, rotation=90)
    plt.xticks(tick_marks, classes, rotation=0)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha='center', va='center')

    plt.show()
