import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
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
    Assign authorship to paragraphs based on Euclidean distances.

    Args:
        document (str): The input document.
        threshold (float): Similarity threshold for determining authorship.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    tfidf_matrix = calculate_paragraph_tfidf(document)

    distances = euclidean_distances(tfidf_matrix)

    num_paragraphs = len(document.split("\n"))
    authorship = np.zeros(num_paragraphs, dtype=int)
    next_author_label = 1

    # Normalize distances and assign authorship
    max_distance = np.max(distances)
    for i in range(num_paragraphs):
        if authorship[i] == 0:
            authorship[i] = next_author_label
            next_author_label += 1

        similar_paragraphs = np.where((authorship == 0))[0]
        normalized_distances = 1 - (distances[i][similar_paragraphs] / max_distance)
        similar_indices = similar_paragraphs[normalized_distances > threshold]
        authorship[similar_indices] = authorship[i]

    return authorship.tolist()

def par2voc(document: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierTFIDFEuclidean:
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):
        predicted_authorship = assign_authorship(document, threshold)
        return predicted_authorship
