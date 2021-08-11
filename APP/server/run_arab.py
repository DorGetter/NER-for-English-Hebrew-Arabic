import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
import torch
import transformers
from transformers import pipeline
import pandas as pd
import json
import os

path = os.getcwd()
parent_path = os.path.dirname(path) #APP/

model = None
tokenizer, label_map = None, None




def look_and_see(text):
    global model
    global tokenizer,label_map


    JsonFile = json.dumps(text)
    with open(str(parent_path) + "/Client/src/RawText.json", "w") as f:
        f.write(JsonFile)
        f.close()


    if model is None and tokenizer is None and label_map is None:
        print("> > > Loading arabic Model...")
        model_name = 'aubmindlab/bert-base-arabertv02'  # The model name to init the model
        def model_init():
            return AutoModelForTokenClassification.from_pretrained(model_name, return_dict=True, num_labels=len(label_map))
        tokenizer , label_map = pd.read_pickle(parent_path + "/server/arabic_model/tokenizer_0_tags_1.pkl")# Extract the tokenizer that saved in pickle file
        model = model_init()
        model.load_state_dict(torch.load(parent_path + "/server/arabic_model/pytorch_model.bin", map_location=torch.device('cpu')))
    


    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence])
    nlp = pipeline('ner', model=model, tokenizer=tokenizer)
    with torch.no_grad():
        output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        output1 = nlp(text)
        new_tokens, new_labels = [], []
        i = 1
        for entity in output1:
            entity['entity'] = label_map[label_indices[0][i]]
            i += 1

        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(label_map.get(label_idx))
                new_tokens.append(token)


        # Extracting the scores
        scores = []
        for entity in output1:
            scores.append(entity['score'])

        # New List to be returned to the model.     
        new_list = []
        for token, label ,score in zip(new_tokens, new_labels ,scores):
            new_list.append({'word':token,'score':str(score) ,'entity':label})

        new_list = new_list[1 :-1]


        jsonFile = json.dumps(new_list)
        with open(str(parent_path) + "/Client/src/textJSON.json", "w") as f:
            f.write(jsonFile)
            f.close()

