import torch
import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertConfig
from transformers import pipeline
from itertools import groupby
import os

path = os.getcwd()
parent_path = os.path.dirname(path) #APP/

model = None
tokenizer, tag_values = None, None


def look_and_see(text):
    global model
    global tokenizer, tag_values


    print()
    JsonFile = json.dumps(text)
    with open(str(parent_path) + "/Client/src/RawText.json", "w") as f:
        f.write(JsonFile)
        f.close()

    if model is None and tokenizer is None and tag_values is None:
        print("> > > Loading hebrew Model...")
        model = torch.load(parent_path + '/server/aleph_model/model_aleph.pth', map_location=torch.device('cpu'))
        tokenizer, tag_values = pd.read_pickle(parent_path + "/server/aleph_model/tokenizer_0_tags_1_aleph.pkl")

    test_sentence = text
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    output1 = nlp(text)
    i = 0
    for entity in output1:
        entity['entity'] = tag_values[label_indices[0][i]]
        i += 1
    
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    scores = []
    for entity in output1:
        scores.append(entity['score'])
    new_list = []
    for token, label, score in zip(new_tokens, new_labels, scores):
        new_list.append({'word': token, 'score': str(score), 'entity': label})
    
    new_list = new_list[1:-1]
    
    to_remove = []
    for i in range(len(new_list)):
        j = new_list[i]
        word_j = j['word']
    
        if (word_j == "'" or word_j == "’"):
            to_remove.append(i)
            to_remove.append(i + 1)
    
        elif (word_j == "."):
            to_remove.append(i)
    
        elif (word_j == '"'):
            to_remove.append(i)
    
        elif (word_j == "-" or word_j == "—"):
            to_remove.append(i)
            to_remove.append(i + 1)
    
        elif (word_j == ","):
            to_remove.append(i)
    
    reduce = 0
    
    for item in to_remove:
        remove = new_list[item - reduce];
        reduce += 1
        new_list.remove(remove)
    
    jsonFile = json.dumps(new_list)
    with open(str(parent_path) + "/Client/src/textJSON.json", "w") as f:
        f.write(jsonFile)
        f.close()
    

    