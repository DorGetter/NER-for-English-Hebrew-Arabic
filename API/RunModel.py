import torch
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertConfig

def run(text):
    model = torch.load('/home/dor/Desktop/NER-Project/Model/model.pth', map_location=torch.device('cpu'))
    tokenizer, tag_values = pd.read_pickle("/home/dor/Desktop/NER-Project/Model/tokenizer_0_tags_1.pkl")

    test_sentence = text

    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    ret_string = ""
    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))
        ret_string+= label +", "+ token+"\n"
    return ret_string
    pass





