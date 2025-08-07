import pandas as pd
import matplotlib.pyplot as plt

# The input string containing the Markdown table


def draw_figure(data_string, xlabel, ylabel, title):
    # a more robust way to parse the markdown table
    # Split the string by lines, remove leading/trailing whitespace, and filter out empty lines and separator lines.
    lines = [line.strip() for line in data_string.strip().split('\n')]
    # get the headers, and remove the spaces
    headers = [header.strip() for header in lines[0].split('|') if header.strip()]
    data = []

    # parse the data
    for line in lines[2:]:
        values = [v.strip() for v in line.split('|') if v.strip()]
        data.append(values)

    # create a dataframe
    df = pd.DataFrame(data, columns=headers)

    # Convert all columns to numeric, handling potential errors
    df = df.apply(pd.to_numeric, errors='coerce')


    # Plotting the data
    plt.figure(figsize=(10, 6))

    for column in df.columns:
        if column != xlabel:
            plt.plot(df[xlabel], df[column], marker='o', linestyle='-', label=column)

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
