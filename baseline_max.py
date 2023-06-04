import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

def calculate_paragraph_frequency(document: str):
    """
    Calculate the word frequency distribution for each paragraph.

    Args:
        document (str): Document containing paragraphs.

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
        document (str): Document containing paragraphs.
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

        max_similarity = 0
        max_similarity_index = -1
        for j in range(i + 1, num_paragraphs):
            if authorship[j] == 0 and similarities[i][j] > threshold:
                if similarities[i][j] > max_similarity:
                    max_similarity = similarities[i][j]
                    max_similarity_index = j

        if max_similarity_index != -1:
            authorship[max_similarity_index] = authorship[i]

    return authorship


def par2voc(document: str):
    """
    Extract vocabulary from a document.

    Args:
        document (str): Document containing paragraphs.

    Returns:
        list: Vocabulary extracted from the document.
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierMax():
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):
        predicted_authorship = assign_authorship(document, threshold)
        return predicted_authorship
