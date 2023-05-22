import torch
import time
import joblib

from util import split_into_sentences
from generate_embeddings import BertEmbeddings
from data import Dataset
from sklearn.ensemble import RandomForestClassifier

def train(dataset):

    embeddings = BertEmbeddings()
    print("Loaded BERT")

    X_docu = []
    y_docu = []    

    X_para = []
    y_para = []

    time_start = time.time()

    for i, instance in enumerate(dataset):
        document = instance.text

        if i == 197:
            pass

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
        #document_embeddings = document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cpu()
            paragraphs_embeddings = paragraphs_embeddings.cpu()

        if torch.isnan(document_embeddings).any():
            print("NaN detected in document")

        X_docu.append(document_embeddings.numpy())
        y_docu.append(instance.multi_author)

        if torch.isnan(paragraphs_embeddings).any():
            print("NaN detected in paragraph")

        X_para.extend(paragraphs_embeddings.numpy())
        y_para.extend(instance.changes)

        if i % 10 == 0:
            print(f"{i}/{len(dataset)}, time elapsed: {(time.time() - time_start)/60} min")


    print("Training docu classifier")
    clf_docu = RandomForestClassifier()
    clf_docu.fit(X_docu, y_docu)
    joblib.dump(clf_docu, "weights/Docu.joblib")

    print("Training para classifier")
    clf_para = RandomForestClassifier()
    clf_para.fit(X_para, y_para)
    joblib.dump(clf_para, "weights/Para.joblib")

    

if __name__=="__main__":
    input_path= 'data/train/'

    dataset = Dataset(input_path)

    train(dataset)