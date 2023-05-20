import os
import json
from collections import namedtuple

Instance = namedtuple("Instance", ["Text", "MultiAuthor", "Changes", "ParagraphAuthors"])

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

    def __getitem__(self, idx):
        return self.instances[idx]


if __name__=="__main__":
    dataset = Dataset()
    print(dataset[0])