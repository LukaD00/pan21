from __future__ import annotations
import os
import json
import joblib

from features import DocumentFeatures, ParagraphFeatures

class Instance:

    def __init__(self, data, multi_author, changes, paragraph_authors, id):
        self.data = data
        self.multi_author = multi_author
        self.changes = changes
        self.paragraph_authors = paragraph_authors
        self.id = id

class RawDataset:

    def __init__(self, path="data/train/"):
        self.instances = []

        for file_name in os.listdir(path):
            if file_name.endswith(".txt"):

                id = file_name[8:-4]
                truth_file_name = f"truth-problem-{id}.json"

                with open(path+file_name, "r") as file:
                    text = file.read()
                
                with open(path+truth_file_name, "r") as file:
                    data = json.load(file)
                    multi_author = data["multi-author"]
                    changes = data["changes"]
                    paragraph_authors = data["paragraph-authors"]

                self.instances.append(Instance(text, multi_author, changes, paragraph_authors, id))

    def __getitem__(self, idx) -> Instance:
        return self.instances[idx]
    
    def __len__(self):
        return len(self.instances)


class DocumentFeaturesDataset:

    def __init__(self, document_feature_extractor : DocumentFeatures, path="data/train/"):
        self.raw_dataset = RawDataset(path)
        self.instances = []
        self.ids = []
        self.X = []
        self.y = []

        for raw_instance in self.raw_dataset:
            
            instance = Instance(
                document_feature_extractor.extract(raw_instance.data),
                raw_instance.multi_author,
                raw_instance.changes,
                raw_instance.paragraph_authors,
                raw_instance.id
            )
            
            self.instances.append(instance)
            self.ids.append(instance.id)
            self.X.append(instance.data)
            self.y.append(instance.multi_author)

    def __getitem__(self, idx) -> Instance:
        return self.instances[idx]
    
    def __len__(self):
        return len(self.instances)
    
    def save(self, path):
        joblib.dump(self, path)

    def load(path) -> DocumentFeaturesDataset:
        return joblib.load(path)



class ParagraphFeaturesDataset:

    def __init__(self, document_feature_extractor : ParagraphFeatures, path="data/train/"):
        self.raw_dataset = RawDataset(path)
        self.instances = []
        self.ids = []
        self.X = []
        self.y = []

        for raw_instance in self.raw_dataset:
            
            instance = Instance(
                document_feature_extractor.extract(raw_instance.data),
                raw_instance.multi_author,
                raw_instance.changes,
                raw_instance.paragraph_authors,
                raw_instance.id
            )
            
            self.instances.append(instance)
            self.ids.append(instance.id)
            self.X.append(instance.data)
            self.y.append(instance.multi_author)

    def __getitem__(self, idx) -> Instance:
        return self.instances[idx]
    
    def __len__(self):
        return len(self.instances)
    
    def save(self, path):
        joblib.dump(self, path)

    def load(path) -> DocumentFeaturesDataset:
        return joblib.load(path)




if __name__=="__main__":
    dataset = RawDataset()
    print(dataset[0])