from matplotlib import pyplot as plt


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

def plot_confusion_matrix(matrix, ax=None, labels=None):
    ax = ax or plt.gca()
    ax.matshow(matrix, cmap='Blues')

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')

    if labels is None:
        ax.set_yticks(range(len(matrix)))
        ax.set_xticks(range(len(matrix)))
    else:
        ax.set_yticks(range(len(matrix)), labels, va='center', rotation=90)
        ax.set_xticks(range(len(matrix)), labels)

    ax.set_ylabel('Actual', weight='bold')
    ax.set_xlabel('Predicted', weight='bold')