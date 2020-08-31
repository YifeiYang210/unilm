import matplotlib.pyplot as plt
import numpy as np
from modeling_layoutlm import LayoutLMForTokenClassification
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer

config1_path = r"./mymodel/config.json"
config2_path = r"C:\Users\63423\Downloads\bert-base-cased\bert-base-cased\config.json"
model1_path = r"./mymodel\pytorch_model.bin"
model2_path = r"C:\Users\63423\Downloads\bert-base-cased\bert-base-cased\pytorch_model.bin"

config1 = BertConfig.from_pretrained(config1_path, num_labels=11)
config2 = BertConfig.from_pretrained(config2_path, num_labels=11)

model1 = LayoutLMForTokenClassification.from_pretrained(model1_path, config=config1)
model2 = BertForTokenClassification.from_pretrained(model2_path, config=config2)


def cal(arr):
    print(np.mean(arr))
    print(np.std(arr))
    print(np.max(arr))
    print(np.min(arr))


a = model1.bert.embeddings.word_embeddings.weight.detach().numpy()
b = model2.bert.embeddings.word_embeddings.weight.detach().numpy()
cal(a)
cal(b)
if a.all() == b.all():
    print(True)
else:
    print(False)
