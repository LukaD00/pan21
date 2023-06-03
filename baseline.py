import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

def calculate_paragraph_frequency(document: str):
    """
    Calculate the word frequency distribution for each paragraph.

    Args:
        paragraphs (list): List of paragraphs.

    Returns:
        numpy.ndarray: 2D array of shape (num_paragraphs, num_features) representing the frequency distribution.
    """
    frequencies = []
    vocabulary = par2voc(document)
    for paragraph in document.split("\n"):
        frequency_vector = np.array([paragraph.count(word) for word in vocabulary])
        frequencies.append(frequency_vector)

    return np.array(frequencies)

def assign_authorship(document, threshold):
    """
    Assign authorship to paragraphs based on cosine similarity.

    Args:
        paragraphs (list): List of paragraphs.
        threshold (float): Similarity threshold for determining authorship.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    frequencies = calculate_paragraph_frequency(document)

    similarities = cosine_similarity(frequencies)

    num_paragraphs = len(document.split("\n"))
    authorship = [0] * num_paragraphs
    next_author_label = 1
    for i in range(num_paragraphs):

        if authorship[i] == 0:
            authorship[i] = next_author_label
            next_author_label += 1

        for j in range(i + 1, num_paragraphs):
            if authorship[j] == 0 and similarities[i][j] > threshold:
                authorship[j] = authorship[i]

    return authorship


def par2voc(document: str):

    nlp = spacy.load('en_core_web_sm')

    doc = nlp(document)

    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifier():

    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):


        predicted_authorship = assign_authorship(document, threshold)

        return predicted_authorship

        # print(predicted_authorship)
        #
        # for i, author in enumerate(predicted_authorship):
        #     print(f"Paragraph {i + 1} has author: {author}")

# document = "The quick brown fox jumps over the lazy dog.\nThe lazy dog is brown.\nThe quick fox jumps over the lazy " \
#              "dog.\nThe brown dog is lazy.\nThe fox jumps over the dog."


