import torch
import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertConfig
from transformers import pipeline
from itertools import groupby

def look_and_see(text):


    JsonFile = json.dumps(text)
    with open("/home/dolev/Downloads/DolevApp/Client/src/RawText.json", "w") as f:
        f.write(JsonFile)
        f.close()


    model = torch.load('/home/dolev/Downloads/DolevApp/server/model.pth', map_location=torch.device('cpu'))
    tokenizer, tag_values = pd.read_pickle("/home/dolev/Downloads/DolevApp/server/tokenizer_0_tags_1.pkl")
    test_sentence = text
    tokenized_sentence =tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    output1 = nlp(text)

    i = 0
    list = []
    words = []
    entities = []
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
    ret_string = ""
    for token, label in zip(new_tokens, new_labels):
        ret_string+= label +" "+ token+"\n"



    ret_string = ret_string[:-1].replace("O ", "")
    ret_string = ret_string.replace("\n", " ")
    str_arr = ret_string.split(" ")
    str_arr = str_arr[1:-1]
    str = " ".join(x for x in str_arr)
    list_tags = [str for str in new_labels if str != 'O']
    scores = []
    for entity in output1:
         scores.append(entity['score'])


    jsonFile = json.dumps(output1)
    with open("/home/dolev/Downloads/DolevApp/Client/src/textJSON.json", "w") as f:
        f.write(jsonFile)
        f.close()

    return output1 
    pass

# def make_zip(text):
#     output , labels , str = run(text)
#     names = []
#     lst_tuple =[]
#     scores = []
#     for entity in output:
#         if entity['entity'] != 'O':
#             names.append(entity['word'])
#             scores.append(entity['score'])
#     lst_tuple = list(zip(labels, names , scores))
#     print(lst_tuple)

#     return lst_tuple
#     pass


# def decode_ner_tags(tag_sequence, tag_probability, non_entity: str = 'O'):
#     """ take tag sequence, return list of entity
#        input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
#        return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
#        """
#     # assert len(tag_sequence) == len(tag_probability)
#     unique_type = list(set(i.split('-')[-1] for i in tag_sequence if i != non_entity))
#     result = []
#     for i in unique_type:
#         mask = [t.split('-')[-1] == i for t, p in zip(tag_sequence, tag_probability)]

#             # find blocks of True in a boolean list
#         group = list(map(lambda x: list(x[1]), groupby(mask)))
#         length = list(map(lambda x: len(x), group))
#         group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]

#         # get entity
#         for g in group_length:
#             result.append([i, g])
#     result = sorted(result, key=lambda x: x[1][0])
#     # print(result)
#     return result
#     pass
