import numpy as np
from rank_bm25 import BM25Okapi
import spacy

def calculate_paragraph_frequency(document: str, bm25_model):
    """
    Calculate the BM25 score distribution for each paragraph.

    Args:
        document (str): The input document.
        bm25_model (BM25Okapi): The BM25 model.

    Returns:
        numpy.ndarray: 1D array of BM25 scores for each paragraph.
    """
    paragraphs = document.split("\n")
    scores = [bm25_model.get_scores(paragraph.lower().split()) for paragraph in paragraphs]
    return np.array(scores)

def assign_authorship(document, threshold):
    """
    Assign authorship to paragraphs based on BM25 similarity.

    Args:
        document (str): The input document.
        threshold (float): Similarity threshold for determining authorship.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    paragraphs = document.split("\n")
    corpus = [paragraph.lower().split() for paragraph in paragraphs]
    bm25_model = BM25Okapi(corpus)

    scores = calculate_paragraph_frequency(document, bm25_model)

    num_paragraphs = len(paragraphs)
    authorship = [0] * num_paragraphs
    next_author_label = 1
    for i in range(num_paragraphs):
        if authorship[i] == 0:
            authorship[i] = next_author_label
            next_author_label += 1

        for j in range(i + 1, num_paragraphs):
            if authorship[j] == 0 and scores[i][j] > threshold:
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

class BaselineClassifierBM25():
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):
        predicted_authorship = assign_authorship(document, threshold)
        return predicted_authorship
