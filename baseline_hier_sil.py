import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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

def assign_authorship(document):
    """
    Assign authorship to paragraphs based on hierarchical clustering.

    Args:
        document (str): The input document.

    Returns:
        list: Predicted authorship labels for each paragraph.
    """
    tfidf_matrix = calculate_paragraph_tfidf(document)

    similarities = cosine_similarity(tfidf_matrix)

    num_paragraphs = len(document.split("\n"))
    authorship = np.zeros(num_paragraphs, dtype=int)

    if num_paragraphs <= 2:
        # Handle case when there are fewer than 2 paragraphs
        authorship[0] = 1  # Assign a default authorship label to all paragraphs
        authorship[1] = 2
        print("Insufficient paragraphs for clustering. Assigning default authorship.")

    else:
        silhouette_scores = []
        for n_clusters in range(2, num_paragraphs):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
            cluster_labels = clustering.fit_predict(similarities)
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                silhouette_scores.append(-1)  # Set a dummy score if less than 2 clusters
            else:
                silhouette_scores.append(silhouette_score(similarities, cluster_labels))

        best_num_clusters = np.argmax(silhouette_scores) + 2

        clustering = AgglomerativeClustering(n_clusters=best_num_clusters, linkage='complete')
        cluster_labels = clustering.fit_predict(similarities)

        unique_labels = np.unique(cluster_labels)
        next_author_label = 1
        for label in unique_labels:
            label_paragraphs = np.where(cluster_labels == label)[0]
            authorship[label_paragraphs] = next_author_label
            next_author_label += 1

        print("Best Silhouette Score:", silhouette_scores[best_num_clusters - 2])

    return authorship.tolist()

def par2voc(document: str):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    vocabulary = set()

    for token in doc:
        if token.is_alpha:
            vocabulary.add(token.text.lower())

    return list(vocabulary)

class BaselineClassifierHierarchicalSil:
    def __init__(self):
        super().__init__()

    def classifyDocument(self, document):
        predicted_authorship = assign_authorship(document)
        return predicted_authorship
