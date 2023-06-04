import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data import *
from features import *
from util import sort_group_names


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


if __name__ == "__main__":

    output_path = 'out/'

    dataset_docu = DocumentFeaturesDataset.load("features/Docu_val_bert.joblib")
    dataset_para = DocumentFeaturesDataset.load("features/Para_val_bert.joblib")

    classifier_docu = joblib.load("classifiers/Docu_Bert_NN_3.joblib")
    classifier_para = joblib.load("classifiers/Para_Bert_NN_3.joblib")

    feature_extractor = DocumentBertEmbeddings()

    solutions = {}

    print("Predicting multi_author labels...")
    for i, instance in enumerate(dataset_docu):
        predicted_multi_author = classifier_docu.predict(instance.data.reshape(1, -1))[0]
        solutions[instance.id] = {"multi-author": predicted_multi_author}
        if i % 100 == 0: print(f"\t{i} / {len(dataset_docu)}")

    print("Predicting changes between paragraphs...")
    for i, instance in enumerate(dataset_para):
        if i % 100 == 0: print(f"\t{i} / {len(dataset_para)}")

        if solutions[instance.id]["multi-author"] == 0:

            document = raw_dataset[i].data
            paragraphs = document.split('\n')
            if paragraphs[-1].strip() == "":
                paragraphs = paragraphs[:-1]
            p = len(paragraphs)

            solutions[instance.id]["changes"] = [0] * (p - 1)
            solutions[instance.id]["paragraph-authors"] = [1] * p
            continue

        predicted_changes = classifier_para.predict(instance.data).astype(np.int32).tolist()
        solutions[instance.id]["changes"] = predicted_changes

        # TASK 3 (ugly, I know)
        raw_dataset = dataset_para.raw_dataset

        document = raw_dataset[i].data
        paragraphs = document.split('\n')
        if paragraphs[-1].strip() == "":
            paragraphs = paragraphs[:-1]
        p = len(paragraphs)

        paragraphs_embeddings = [feature_extractor.extract(p) for p in paragraphs]

        indexes_to_combine = []
        current = [0]
        for p_index in range(len(predicted_changes)):
            if predicted_changes[p_index] == 0:
                current.append(p_index + 1)
            else:
                indexes_to_combine.append(current)
                current = [p_index + 1]
        indexes_to_combine.append(current)

        if len(indexes_to_combine) <= 1:
            solutions[instance.id]["paragraph-authors"] = [1] * p
            continue

        paragraph_embeddings_combined = []
        for index_set in indexes_to_combine:
            paragraph_set = torch.zeros(768)
            for index in index_set:
                paragraph_set += paragraphs_embeddings[index]
            paragraph_set /= len(index_set)
            paragraph_embeddings_combined.append(paragraph_set.numpy())

        best_silhoutte = None
        best_attribution = None
        for k in [2, 3, 4, 5]:
            if k >= len(paragraph_embeddings_combined):
                break

            dimensionality_reduction = PCA(n_components=2)
            paragraph_embeddings_combined = dimensionality_reduction.fit_transform(paragraph_embeddings_combined)

            clusterer = KMeans(n_clusters=k, n_init=5)
            # clusterer = AgglomerativeClustering(n_clusters=k)
            predictions = clusterer.fit_predict(paragraph_embeddings_combined)
            score = silhouette_score(paragraph_embeddings_combined, predictions)

            if best_silhoutte is None or score > best_silhoutte:
                best_silhoutte = score
                best_attribution = predictions

        if len(paragraph_embeddings_combined) == 2:
            best_attribution = [0, 1]

        final_attribution = []
        for i in range(len(best_attribution)):
            final_attribution += [best_attribution[i]] * len(indexes_to_combine[i])
        final_attribution = sort_group_names(final_attribution)

        solutions[instance.id]["paragraph-authors"] = final_attribution

    print(f"Saving results to {output_path}")
    for id in solutions:
        file_name = output_path + 'solution-problem-' + id + '.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solutions[id], file_handle, default=myconverter)
