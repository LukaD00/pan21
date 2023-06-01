import joblib
import numpy as np

from data import *
from features import *


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
    classifier_para = joblib.load("classifiers/Para_Bert_NN_1.joblib")

    solutions = {}

    print("Predicting multi_author labels...")
    for i, instance in enumerate(dataset_docu):
        predicted_multi_author = classifier_docu.predict(instance.data.reshape(1, -1))[0]
        solutions[instance.id] = {"multi-author": predicted_multi_author}
        if i % 100 == 0: print(f"\t{i} / {len(dataset_docu)}")

    print("Predicting changes between paragraphs...")
    for i, instance in enumerate(dataset_para):
        predicted_changes = classifier_para.predict(instance.data).astype(np.int32).tolist()
        solutions[instance.id]["changes"] = predicted_changes
        if i % 100 == 0: print(f"\t{i} / {len(dataset_para)}")

    print(f"Saving results to {output_path}")
    for id in solutions:
        file_name = output_path+'solution-problem-'+id+'.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solutions[id], file_handle, default=myconverter)