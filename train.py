import joblib

from data import *
from features import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


    

if __name__=="__main__":

    #features = DocumentBertEmbeddings()
    #dataset = DocumentFeaturesDataset(features)

    dataset = DocumentFeaturesDataset.load("features/Docu_train_bert.joblib")
    classifier = RandomForestClassifier(1800)
    save_file = "classifiers/Docu_Bert_RF_1800"

    classifier.fit(dataset.X, dataset.y)
    joblib.dump(classifier, save_file)


   
        