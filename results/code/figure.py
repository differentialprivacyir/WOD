import pandas as pd
import matplotlib.pyplot as plt

# pip install arabic-reshaper python-bidi -i https://mirror-pypi.runflare.com/simple
import arabic_reshaper
from bidi.algorithm import get_display

markers = [
    's', 'o', 'v', '^', 'D', '*', '>', '<'
]

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

    for index, column in enumerate(df.columns):
        if column != xlabel:
            plt.plot(df[xlabel], df[column], marker=markers[index], linestyle='-', label=column)

    # Adding labels and title
    plt.xlabel(get_display(arabic_reshaper.reshape(xlabel)))
    plt.ylabel(get_display(arabic_reshaper.reshape(ylabel)))
    plt.title(get_display(arabic_reshaper.reshape(title)))
    plt.legend()
    plt.grid(True)
    plt.show()
