from data import *
import random
from util import sort_group_names, clusters_to_changes
import numpy as np

solutions = {}
output_path = 'out/' 
dataset = RawDataset(path="data/validation/")

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

for i, instance in enumerate(dataset):  
    document = instance.data
    paragraphs = document.split('\n')
    if paragraphs[-1].strip() == "":
        paragraphs = paragraphs[:-1]
    paragraph_count = len(paragraphs)

    num_authors = random.randint(1,4)
    
    solutions[instance.id] = {}
    solutions[instance.id]["paragraph-authors"] = sort_group_names([random.randint(0,num_authors-1) for _ in range(paragraph_count)])
    solutions[instance.id]["changes"] = clusters_to_changes(solutions[instance.id]["paragraph-authors"])
    solutions[instance.id]["multi-author"] = 1 if num_authors>1 else 0
    
    if i % 100 == 0: print(f"\t{i} / {len(dataset)}")

print(f"Saving results to {output_path}")
for id in solutions:
    file_name = output_path+'solution-problem-'+id+'.json'
    with open(file_name, 'w') as file_handle:
        json.dump(solutions[id], file_handle, default=myconverter)