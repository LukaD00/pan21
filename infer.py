import torch
import json
import numpy as np
import joblib
import glob

from util import split_into_sentences
from generate_embeddings import BertEmbeddings

def infer(corpora, inputpath, outputpath):

    embeddings = BertEmbeddings()

    with open('weights/Docu.joblib', 'rb') as file_handle:
        clf_docu = joblib.load(file_handle)

    with open('weights/Para.joblib', 'rb') as file_handle:
        clf_para = joblib.load(file_handle)

    for document_path in corpora:
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        document_id = document_path[len(inputpath)+9:-4]

        if not document or not document_id:
            continue

        document_embeddings = torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cuda()

        sentence_count = 0
        paragraphs_embeddings = []
        paragraphs = document.split('\n')

        previous_para_embeddings = None
        previous_para_length = None

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)

            current_para_embeddings = torch.zeros(768)
            if torch.cuda.is_available():
                current_para_embeddings = current_para_embeddings.cuda()

            current_para_length = len(sentences)

            for sentence in sentences:
                sentence_count += 1
                sentence_embedding = embeddings.generate_sentence_embedding(sentence)
                current_para_embeddings.add_(sentence_embedding)
                document_embeddings.add_(sentence_embedding)
                del sentence_embedding, sentence

            if previous_para_embeddings is not None:
                two_para_lengths = previous_para_length + current_para_length
                two_para_embeddings = (
                    previous_para_embeddings + current_para_embeddings)/two_para_lengths

                paragraphs_embeddings.append(two_para_embeddings)

            previous_para_embeddings = current_para_embeddings
            previous_para_length = current_para_length
            del sentences
            del paragraph

        del previous_para_embeddings, previous_para_length
        del current_para_embeddings, current_para_length
        del two_para_embeddings

        paragraphs_embeddings = torch.stack(paragraphs_embeddings, dim=0)
        document_embeddings = document_embeddings/sentence_count
        document_embeddings = document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cpu()
            paragraphs_embeddings = paragraphs_embeddings.cpu()

        # PREDICTIONS

        try:
            document_label = clf_docu.predict(document_embeddings)
        except:
            # print('in except docu')
            document_label = [0]

        try:
            paragraphs_labels = clf_para.predict(paragraphs_embeddings)
        except:
            # print('in except para')
            paragraphs_labels = np.zeros(len(paragraphs)-1)
        paragraphs_labels = paragraphs_labels.astype(np.int32)

        solution = {
            'multi-author': document_label[0],
            'changes': paragraphs_labels.tolist()
        }

        file_name = outputpath+'/solution-problem-'+document_id+'.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)
        print(f"Solved {document_id}: {solution}")

        del document_embeddings, document_label
        del paragraphs_embeddings, paragraphs_labels
        del solution
        del document, document_id
        del paragraphs

    del clf_docu, clf_para


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


if __name__=="__main__":
    input_path= 'data/validation'
    output_path = 'out' 

    dataset = glob.glob(input_path+'/*.txt')

    infer(dataset, input_path, output_path)