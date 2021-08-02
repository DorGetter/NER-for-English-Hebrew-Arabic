import torch
import numpy as np
import pandas as pd
import json
from transformers import BertTokenizer, BertConfig
from transformers import pipeline
from itertools import groupby

# def look_and_see(text):


#     JsonFile = json.dumps(text)
#     with open("/home/dor/Downloads/DolevApp/Client/src/RawText.json", "w") as f:
#         f.write(JsonFile)
#         f.close()


#     model = torch.load('/home/dor/Downloads/DolevApp/server/model.pth', map_location=torch.device('cpu'))
#     tokenizer, tag_values = pd.read_pickle("/home/dor/Downloads/DolevApp/server/tokenizer_0_tags_1.pkl")



#     test_sentence = text
#     tokenized_sentence =tokenizer.encode(test_sentence)
#     input_ids = torch.tensor([tokenized_sentence])
#     with torch.no_grad():
#         output = model(input_ids)
#     label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
#     tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
#     nlp = pipeline('ner', model=model, tokenizer=tokenizer)
#     output1 = nlp(text)

#     i = 0
#     list = []
#     words = []
#     entities = []
#     for entity in output1:
#         entity['entity'] = tag_values[label_indices[0][i]]
#         i += 1
   
#     new_tokens, new_labels = [], []
#     for token, label_idx in zip(tokens, label_indices[0]):
#         if token.startswith("##"):
#             new_tokens[-1] = new_tokens[-1] + token[2:]
#         else:
#             new_labels.append(tag_values[label_idx])
#             new_tokens.append(token)
#     ret_string = ""
#     for token, label in zip(new_tokens, new_labels):
#         ret_string+= label +" "+ token+"\n"



    
#     ret_string = ret_string[:-1].replace("O ", "")
#     ret_string = ret_string.replace("\n", " ")
#     str_arr = ret_string.split(" ")
#     str_arr = str_arr[1:-1]
#     str = " ".join(x for x in str_arr)
#     list_tags = [str for str in new_labels if str != 'O']
#     scores = []
#     for entity in output1:
#          scores.append(entity['score'])

#     for item in output1:
        
#         word = item['word']
#         if word != None:
#             if word[0] == '#':
#                 output1.remove(item)
#             if word[0] == '[':
#                 output1.remove(item)
#             if word[0] == '[SEP]':
#                 output1.remove(item)
                




    
#     jsonFile = json.dumps(output1)
#     with open("/home/dor/Downloads/DolevApp/Client/src/textJSON.json", "w") as f:
#         f.write(jsonFile)
#         f.close()

#     return output1 
#     pass



def look_and_see(text):


    JsonFile = json.dumps(text)
    with open("/home/dor/Downloads/DolevApp/Client/src/RawText.json", "w") as f:
        f.write(JsonFile)
        f.close()


    model = torch.load('/home/dor/Downloads/DolevApp/server/model.pth', map_location=torch.device('cpu'))
    tokenizer, tag_values = pd.read_pickle("/home/dor/Downloads/DolevApp/server/tokenizer_0_tags_1.pkl")


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
    scores = []
    for entity in output1:
         scores.append(entity['score'])
    new_list = []
    for token, label ,score in zip(new_tokens, new_labels ,scores):
        new_list.append({'word':token,'score':score ,'entity':label})



    jsonFile = json.dumps(new_list[1:-1])
    with open("/home/dor/Downloads/DolevApp/Client/src/textJSON.json", "w") as f:
        f.write(jsonFile)
        f.close()

    return output1 