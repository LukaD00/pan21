import joblib

from data import *
from features import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def train_task1():

    #features = DocumentBertEmbeddings()
    #dataset = DocumentFeaturesDataset(features)

    dataset = DocumentFeaturesDataset.load("features/Docu_train_bert.joblib")
    classifier = RandomForestClassifier(1800)
    save_file = "classifiers/Docu_Bert_RF_1800"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)


def train_task2():
    dataset = ParagraphFeaturesDataset.load("features/Para_train_bert.joblib")
    classifier = MLPClassifier([50,10,10], max_iter=300)
    save_file = "classifiers/Para_Bert_NN_3.joblib"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)
    

def train_task3_kmeans():
    # Very messy, I know


    task2_classifier = joblib.load("classifiers/Para_Bert_NN_1.joblib")
    dataset = ParagraphFeaturesDataset.load("features/Para_train_bert.joblib")
    raw_dataset = dataset.raw_dataset
    feature_extractor = DocumentBertEmbeddings()

    X = []
    #for instance in dataset:

    load_features = False

    if load_features:
        pass
    else:
        i = 2
        instance = dataset[i]
        document = raw_dataset[i].data
        paragraphs = document.split('\n')
        if paragraphs[-1].strip() == "":
            paragraphs = paragraphs[:-1]
        changes = task2_classifier.predict(instance.data)
        print(changes)
        print(len(paragraphs))
        
        paragraphs_embeddings = [feature_extractor.extract(p) for p in paragraphs]
        print(len(paragraphs_embeddings))
    
        indexes_to_combine = []
        current = [0]
        for p_index in range(len(changes)):
            if changes[p_index] == 0:
                current.append(p_index+1)
            else:
                indexes_to_combine.append(current)
                current = [p_index+1]
        indexes_to_combine.append(current)
        print(indexes_to_combine)

        paragraph_embeddings_combined = []
        for index_set in indexes_to_combine:
            paragraph_set = torch.zeros(768)
            for index in index_set:
                paragraph_set += paragraphs_embeddings[index]
            paragraph_set /= len(index_set)
            paragraph_embeddings_combined.append(paragraph_set)

        print(len(paragraph_embeddings_combined))

if __name__=="__main__":

    #train_task2()

    train_task3_kmeans()

    


   
        