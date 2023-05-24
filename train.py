import joblib

from generate_embeddings import BertEmbeddings
from data import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train(dataset, clf_docu_save_path, clf_para_save_path, calculate_embeddings=True):

    if calculate_embeddings:
        embeddings = BertEmbeddings()
        print("Loaded BERT")
        X_docu, y_docu, X_para, y_para = embeddings.generate_docu_para_embeddings(dataset)
    
        joblib.dump(X_docu, "weights/X_docu_train.joblib")
        joblib.dump(y_docu, "weights/y_docu_train.joblib")
        joblib.dump(X_para, "weights/X_para_train.joblib") 
        joblib.dump(y_para, "weights/y_para_train.joblib")

    else:
        with open('weights/X_docu_train.joblib', 'rb') as file_handle:
            X_docu = joblib.load(file_handle)
        with open('weights/y_docu_train.joblib', 'rb') as file_handle:
            y_docu = joblib.load(file_handle)
        with open('weights/X_para_train.joblib', 'rb') as file_handle:
            X_para = joblib.load(file_handle)
        with open('weights/y_para_train.joblib', 'rb') as file_handle:
            y_para = joblib.load(file_handle)

    print("Training docu classifier")
    #clf_docu = RandomForestClassifier(n_estimators=1800, criterion="gini", min_samples_leaf=1, min_samples_split=2)
    clf_docu = MLPClassifier([50, 10])
    clf_docu.fit(X_docu, y_docu)
    joblib.dump(clf_docu, clf_docu_save_path)

    print("Training para classifier")
    #clf_para = RandomForestClassifier(n_estimators=250, criterion="gini", min_samples_leaf=1, min_samples_split=2)
    clf_para = MLPClassifier([50, 10])
    clf_para.fit(X_para, y_para)
    joblib.dump(clf_para, clf_para_save_path)

    

if __name__=="__main__":
    input_path= "data/train/"
    clf_docu_save_path = "weights/Docu.joblib"
    clf_para_save_path = "weights/Para.joblib"

    dataset = Dataset(input_path)
    train(dataset, clf_docu_save_path, clf_para_save_path, calculate_embeddings=True)