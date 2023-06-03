from baseline import BaselineClassifier
from baseline_tfidf import BaselineClassifierTFIDF
from baseline_tfidf_jaccard import BaselineClassifierTDIDFJaccard
from baseline_bm25 import BaselineClassifierBM25
from baseline_euclidean import BaselineClassifierEuclidean
from baseline_tfidf_euclidean import BaselineClassifierTFIDFEuclidean

import os
import json


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content


if __name__ == '__main__':
    DATA_DIR = 'data/validation'
    OUTPUT_DIR = 'data/solutions_baseline'
    #model = BaselineClassifier()
    model = BaselineClassifierTFIDF()
    #model = BaselineClassifierTDIDFJaccard()
    #model = BaselineClassifierBM25()
    #model = BaselineClassifierTFIDFEuclidean()

    for docindex in range(1, (len([name for name in os.listdir(DATA_DIR) if
                                   os.path.isfile(os.path.join(DATA_DIR, name))]) // 2) + 1):
        print(docindex)
        document_name = "problem-" + str(docindex) + ".txt"
        document_path = os.path.join(DATA_DIR, document_name)
        document = read_file(document_path)

        authorship_list = model.classifyDocument(document, threshold=0.15)
        changes = [1 if authorship_list[i] != authorship_list[i+1] else 0 for i in range(len(authorship_list)-1)]

        solution = {
            "multi-author": 1 if len(set(authorship_list)) > 1 else 0,
            "changes": changes,
            "paragraph-authors": authorship_list
        }

        output_filename = "solution-problem-" + str(docindex) + ".json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(solution, output_file, indent=0)
