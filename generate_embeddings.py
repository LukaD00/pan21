import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import time
from util import split_into_sentences

class BertEmbeddings:


    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


    def generate_sentence_embedding(self, sentence):
        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_sentence = self.tokenizer.tokenize(marked_sentence)
        if len(tokenized_sentence) > 512:  # truncate the sentence if it is longer than 512
            tokenized_sentence = tokenized_sentence[:512]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        segment_ids = [1] * len(tokenized_sentence)

        token_tensor = torch.tensor([indexed_tokens])
        segment_tensor = torch.tensor([segment_ids])

        if torch.cuda.is_available():
            token_tensor = token_tensor.cuda()
            segment_tensor = segment_tensor.cuda()

        with torch.no_grad():
            encoded_layers, _ = self.model(token_tensor, segment_tensor)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = torch.sum(token_embeddings[-4:, :, :], dim=0)
        sentence_embedding_sum = torch.sum(token_embeddings, dim=0)

        del marked_sentence
        del tokenized_sentence
        del indexed_tokens, segment_ids
        del token_tensor
        del segment_tensor
        del encoded_layers
        del token_embeddings

        return sentence_embedding_sum
    

    def generate_docu_para_embeddings(self, dataset):
        X_docu = []
        y_docu = []    

        X_para = []
        y_para = []

        time_start = time.time()

        for i, instance in enumerate(dataset):
            document = instance.text

            #if i == 197:
            #    pass

            document_embeddings = torch.zeros(768)

            if torch.cuda.is_available():
                document_embeddings = document_embeddings.cuda()

            sentence_count = 0
            paragraphs_embeddings = []
            paragraphs = document.split('\n')
            if paragraphs[-1].strip() == "":
                paragraphs = paragraphs[:-1]

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
                    sentence_embedding = self.generate_sentence_embedding(sentence)
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
                print("WARNING: NaN detected in document")

            X_docu.append(document_embeddings.numpy())
            y_docu.append(instance.multi_author)

            if torch.isnan(paragraphs_embeddings).any():
                print("WARNING: NaN detected in paragraph")

            X_para.extend(paragraphs_embeddings.numpy())
            y_para.extend(instance.changes)

            if i % 10 == 0:
                print(f"{i}/{len(dataset)}, time elapsed: {(time.time() - time_start)/60} min")

        return X_docu, y_docu, X_para, y_para