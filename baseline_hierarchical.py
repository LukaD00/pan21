import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
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
    Assign authorship to paragraphs based on hierarchical clustering.

    Args:
        document (str): The input document.
        threshold (float): Similarity threshold for determining authorship.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    tfidf_matrix = calculate_paragraph_tfidf(document)

    similarities = cosine_similarity(tfidf_matrix)

    num_paragraphs = len(document.split("\n"))
    authorship = np.zeros(num_paragraphs, dtype=int)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='ward')
    clustering.fit(similarities)

    cluster_labels = clustering.labels_
    unique_labels = np.unique(cluster_labels)

    next_author_label = 1
    for label in unique_labels:
        label_paragraphs = np.where(cluster_labels == label)[0]
        authorship[label_paragraphs] = next_author_label
        next_author_label += 1

    mean_distance = np.mean(clustering.distances_)

    print("Mean Distance:", mean_distance)

    return authorship.tolist()

def par2voc(document: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierHierarchical:
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document, threshold=0.7):
        predicted_authorship = assign_authorship(document, threshold)
        return predicted_authorship
