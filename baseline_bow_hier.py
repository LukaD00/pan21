import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import cosine_similarity
import spacy

def calculate_paragraph_frequency(document: str):
    """
    Calculate the word frequency distribution for each paragraph.

    Args:
        document (str): The input document.

    Returns:
        numpy.ndarray: 2D array of shape (num_paragraphs, num_features) representing the frequency distribution.
    """
    frequencies = []
    vocabulary = par2voc(document)
    for paragraph in document.split("\n"):
        frequency_vector = np.array([paragraph.count(word) for word in vocabulary])
        frequencies.append(frequency_vector)

    return np.array(frequencies)

def assign_authorship(document):
    """
    Assign authorship to paragraphs based on agglomerative clustering with silhouette scores.

    Args:
        document (str): The input document.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    frequencies = calculate_paragraph_frequency(document)

    similarities = cosine_similarity(frequencies)

    num_paragraphs = len(document.split("\n"))
    authorship = [0] * num_paragraphs

    if num_paragraphs == 2:
        # Handle case when there are fewer than 2 paragraphs
        authorship[0] = 1  # Assign a default authorship label to all paragraphs
        authorship[1] = 2
        print("Insufficient paragraphs for clustering. Assigning default authorship.")

    else:
        silhouette_scores = []
        for n_clusters in range(2, num_paragraphs):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(similarities)
            silhouette_scores.append(silhouette_score(similarities, cluster_labels))

        best_num_clusters = np.argmax(silhouette_scores) + 2

        clustering = AgglomerativeClustering(n_clusters=best_num_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(similarities)

        unique_labels = np.unique(cluster_labels)
        next_author_label = 1
        for label in unique_labels:
            label_paragraphs = np.where(cluster_labels == label)[0].tolist()
            for paragraph_idx in label_paragraphs:
                authorship[paragraph_idx] = next_author_label
            next_author_label += 1

        print("Best Silhouette Score:", silhouette_scores[best_num_clusters - 2])

    return authorship


def par2voc(document: str):

    nlp = spacy.load('en_core_web_sm')

    doc = nlp(document)

    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierBH:

    def __init__(self):
        super().__init__()

    def classifyDocument(self, document):
        predicted_authorship = assign_authorship(document)
        return predicted_authorship
