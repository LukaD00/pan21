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
    classifier = RandomForestClassifier(1800)
    save_file = "classifiers/Para_Bert_RF_1800"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)
    

def train_task3_kmeans():
    task2_classifier = joblib.load("classifiers/Docu_Bert_NN_1")
    dataset = ParagraphFeaturesDataset.load("features/Para_train_bert.joblib")

    X = []
    #for instance in dataset:
    instance = dataset[0]
    print(instance.data)



if __name__=="__main__":

    train_task3_kmeans()

    


   
        