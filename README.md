# NER_Project in English , Hebrew , and Arabic
-----------------

## Abstract

In any text content, there are some terms that are more informative and unique in context. Named Entity Recognition (NER) also known as information extraction/chunking is the process in which algorithm extracts the real world noun entity from the text data and classifies them into predefined categories like person, place, time, organization, etc.

## BERT MODEL

In our project we fine-tuned BERT models to perform named-entity 
recognition (NER) in three languages (English, Hebrew and Arabic).
The project provides web-based interface (written in REACT) that allows 
the user to enter text in one of the languages and get an output of the text
labeled into entities, and a table that shows for each word in the text what is 
the entity, and the confidence score.</b>












You will find this article so useful https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
![](https://github.com/DorGetter/NER_Project/blob/main/bert-input-output.png)


## Installation

### With pip
This repository is tested on Python 3.6+, Flax 0.3.2+, PyTorch 1.3.1+ and TensorFlow 2.3+, transformers , seqeval ,Flask ,and BS4.

### TRANSFORMERS
```bash
pip install transformers
```
----------------
### FLASK
```bash
pip install flask
```
----------------
### SEQEVAL
```bash
pip install seqeval
```
----------------


## Training The Model With Google Colab
![](https://github.com/DorGetter/NER_Project/blob/main/GoogleColabTrain.png)

