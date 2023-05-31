import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from util import split_into_sentences

class DocumentFeatures:

    """
    Superclass for document-level features for task 1.
    """

    def extract(self, document : str):
        pass


class ParagraphFeatures:

    """
    Superclass for document-level features for task 1.
    """

    def extract(self, document : str):
        pass


class DocumentBertEmbeddings(DocumentFeatures):

    def __init__(self, file):
        super().__init__(file)
        self.model = BertModel.from_pretrained('bert-base-cased')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        

    def extract(self, document : str):
        document_embeddings = torch.zeros(768)

        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cuda()

        sentence_count = 0
        paragraphs = document.split('\n')
        if paragraphs[-1].strip() == "":
            paragraphs = paragraphs[:-1]

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)

            for sentence in sentences:
                sentence_count += 1
                sentence_embedding = self.generate_sentence_embedding(sentence)
                document_embeddings.add_(sentence_embedding)

        document_embeddings = document_embeddings/sentence_count
        #document_embeddings = document_embeddings.unsqueeze(0)

        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cpu()

        if torch.isnan(document_embeddings).any():
            print("WARNING: NaN detected in document")

        return document_embeddings


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

        return sentence_embedding_sum