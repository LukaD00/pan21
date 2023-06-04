import re

import numpy as np
from matplotlib import pyplot as plt

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "e.g." in text: text = text.replace("e.g.", "e<prd>g<prd>")
    if "..." in text: text = text.replace("...", "<prd><prd><prd>")
    if "i.e." in text: text = text.replace("i.e.", "i<prd>e<prd>")
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != '']
    return sentences


def sort_group_names(grouping):
    mapping = {}
    occured = []
    current = 1
    for x in grouping:
        if x not in occured:
            occured.append(x)
            mapping[x] = current
            current += 1
    return [mapping[x] for x in grouping]


def clusters_to_changes(grouping):
    changes = []
    for i in range(len(grouping) - 1):
        changes.append(grouping[i] == grouping[i + 1])
    return changes


def count_paragraphs(documents):
    paragraphs_cnt = []
    for instance in documents:
        paragraphs = instance.data.split('\n')
        if paragraphs[-1].strip() == "":
            paragraphs = paragraphs[:-1]

        paragraphs_cnt.append(len(paragraphs))
    return paragraphs_cnt

def calculate_data_statistics(data, name):
    # Calculate mean, median, minimum, and maximum
    mean_value = np.mean(data)
    median_value = np.median(data)
    min_value = np.min(data)
    max_value = np.max(data)

    # Print the calculated values
    print("[" + name + "] Mean:", mean_value)
    print("[" + name + "] Median:", median_value)
    print("[" + name + "] Minimum:", min_value)
    print("[" + name + "] Maximum:", max_value)

def plot_paragraphs_distributions(paragraphs1, paragraphs2):
    # Plotting the histograms
    plt.hist(paragraphs1, bins=range(min(paragraphs1), max(paragraphs1) + 2),
             align='left', edgecolor='blue', alpha=0.5, label='Train')
    plt.hist(paragraphs2, bins=range(min(paragraphs2), max(paragraphs2) + 2),
             align='left', edgecolor='orange', alpha=0.5, label='Validation')

    # Adding labels and title
    plt.xlabel('Number of Paragraphs')
    plt.ylabel('Frequency')
    plt.title('Distribution of Paragraphs across train and validation data sets')

    # Adding legend
    plt.legend()

    # Display the graph
    # plt.show()

    # Save the histogram
    plt.savefig('figures/paragraph_stats.png')


if __name__ == "__main__":
    grouping = [1, 1, 5, 5, 3, 2, 1, 5]
    print(sort_group_names(grouping))
