# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class USEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                 hidden_size=384):
        super(USEncoder, self).__init__()
        # 加载预训练的Universal Sentence Encoder模型和分词器 embeddings.shape == torch.Size([1, 384])
        self._hidden_size = hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # embeddings.shape == torch.Size([1, 768])
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')  # NOQA
        # model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # model = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, sentences):
        tokens = self.tokenizer(sentences, padding=True,
                                truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


if __name__ == '__main__':
    sentences = ["Pick apple from top drawer and place on counter."]
    USE_model = USEncoder()
    embeddings = USE_model(sentences)
    print(embeddings.shape)
