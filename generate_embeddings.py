import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


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