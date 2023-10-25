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