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
    classifier = MLPClassifier([50,20,10], max_iter=200, alpha=0.0001, batch_size=32, verbose=True)
    save_file = "classifiers/Docu_Bert_NN_4.joblib"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)


def train_task2():
    dataset = ParagraphFeaturesDataset.load("features/Para_train_bert.joblib")
    classifier = MLPClassifier([50,10,10], max_iter=300, alpha=0.001, batch_size=32, verbose=True)
    save_file = "classifiers/Para_Bert_NN_5.joblib"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)
    

if __name__=="__main__":

    train_task1()
    train_task2()



    


   
        