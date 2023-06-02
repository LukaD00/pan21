import joblib
import numpy as np

from data import *
from features import *
from util import sort_group_names, clusters_to_changes

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


if __name__=="__main__":

    output_path = 'out/' 

    dataset_docu = DocumentFeaturesDataset.load("features/Docu_val_bert.joblib")
    dataset_para = DocumentFeaturesDataset.load("features/Para_val_bert.joblib")

    classifier_docu = joblib.load("classifiers/Docu_Bert_NN_1.joblib")

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

            solutions[instance.id]["changes"] = [0] * (p-1)
            solutions[instance.id]["paragraph-authors"] = [1] * p
            continue


        # TASK 3 (ugly, I know)
        raw_dataset = dataset_para.raw_dataset
        
        document = raw_dataset[i].data
        paragraphs = document.split('\n')
        if paragraphs[-1].strip() == "":
            paragraphs = paragraphs[:-1]
        p = len(paragraphs)
        
        paragraphs_embeddings = [feature_extractor.extract(p) for p in paragraphs]

        best_silhoutte = None
        best_attribution = None
        for k in [2,3,4,5]:
            if k >= len(paragraphs_embeddings):
                break

            dimensionality_reduction = PCA(n_components=2)
            paragraphs_embeddings = dimensionality_reduction.fit_transform(paragraphs_embeddings)

            clusterer = KMeans(n_clusters=k, n_init=5)
            #clusterer = AgglomerativeClustering(n_clusters=k)
            predictions = clusterer.fit_predict(paragraphs_embeddings)
            score = silhouette_score(paragraphs_embeddings, predictions)

            if best_silhoutte is None or score > best_silhoutte:
                best_silhoutte = score
                best_attribution = predictions

        if len(paragraphs_embeddings) == 2:
            best_attribution = [0,1]

        final_attribution = sort_group_names(best_attribution)

        solutions[instance.id]["paragraph-authors"] = final_attribution
        solutions[instance.id]["changes"] = clusters_to_changes(final_attribution)
        






    print(f"Saving results to {output_path}")
    for id in solutions:
        file_name = output_path+'solution-problem-'+id+'.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solutions[id], file_handle, default=myconverter)

