import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
import spacy

def calculate_paragraph_tfidf(document: str):
    """
    Calculate the TF-IDF representation for each paragraph.

    Args:
        document (str): The input document.

    Returns:
        numpy.ndarray: 2D array of shape (num_paragraphs, num_features) representing the TF-IDF representation.
    """
    paragraphs = document.split("\n")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs)

    return tfidf_matrix.toarray()

def assign_authorship(document, threshold):
    """
    Assign authorship to paragraphs based on Jaccard similarity.

    Args:
        document (str): The input document.
        threshold (float): Similarity threshold for determining authorship.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    tfidf_matrix = calculate_paragraph_tfidf(document)

    num_paragraphs = len(document.split("\n"))
    authorship = np.zeros(num_paragraphs, dtype=int)
    next_author_label = 1
    for i in range(num_paragraphs):
        if authorship[i] == 0:
            authorship[i] = next_author_label
            next_author_label += 1

        for j in range(i + 1, num_paragraphs):
            if authorship[j] == 0:
                similarity = jaccard_score(tfidf_matrix[i] > 0, tfidf_matrix[j] > 0)
                if similarity > threshold:
                    authorship[j] = authorship[i]

    return authorship.tolist()

def par2voc(document: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierTDIDFJaccard:
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):
        predicted_authorship = assign_authorship(document, threshold)
        return predicted_authorship
