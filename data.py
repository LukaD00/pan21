import os
import json
from collections import namedtuple

class Instance:

    def __init__(self, text, multi_author, changes, paragraph_authors):
        self.text = text
        self.multi_author = multi_author
        self.changes = changes
        self.paragraph_authors = paragraph_authors
        

class Dataset:

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

                self.instances.append(Instance(text, multi_author, changes, paragraph_authors))

    def __getitem__(self, idx) -> Instance:
        return self.instances[idx]
    
    def __len__(self):
        return len(self.instances)


if __name__=="__main__":
    dataset = Dataset()
    print(dataset[0])