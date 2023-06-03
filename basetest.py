from baseline import BaselineClassifier

import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    return file_content

if __name__ == '__main__':


    TRAIN_DIR= 'data/validation'
    model = BaselineClassifier()
    for docindex in range(1, (len([name for name in os.listdir(TRAIN_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, name))]) // 2) + 1):
        document_name = "problem-"+str(docindex)+".txt"
        document_path = os.path.join(TRAIN_DIR, document_name)
        document = read_file(document_path)

        if docindex < 10:
            authorship_list = model.classifyDocument(document)
            print(authorship_list)
