# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:00:33 2024

@author: gm205
"""

# import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizer

class NERModel:
    def __init__(self):
        # if(torch.cuda.is_available()):
        #     self.device = torch.device("cuda:0")
        # else:
        #     self.device = torch.device("cpu:0")

        # DistilBert Tokenizer & Model for NER
        print("Loading DistilBert Tokenizer & Model for NER.")
        ner_model_name = "Raj-sharma01/results"
        # ner_model_name = "Gkumi/fresh-model-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(ner_model_name)
        # self.model = DistilBertForTokenClassification.from_pretrained(ner_model_name).to(self.device)
        self.model = DistilBertForTokenClassification.from_pretrained(ner_model_name)

    def predict_entities(self, text):
        # inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(self.device)
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        # predictions = torch.argmax(outputs.logits, dim=2)
        import numpy as np
        predictions = outputs.logits.detach().numpy().argmax(axis=2)
        
        # Convert the input_ids back to words
        input_words = [self.tokenizer.decode([id]) for id in inputs["input_ids"].tolist()[0]]
        
        # Convert the predictions to entity labels
        predicted_labels = [self.model.config.id2label[id] for id in predictions[0].tolist()]
        
        # Pair the words with their predicted entities
        entities = [(word, label) for word, label in zip(input_words, predicted_labels)]
        
        # Initialize an empty list for formatted output
        formatted_output = []
        
        # Iterate over words and their predicted labels
        for word, label in entities:
            if word in ["[CLS]", "[SEP]"]:
                # Ignore special tokens
                continue
            elif word.startswith("##"):
                # This is a word piece, remove '##' and append it to the last word
                formatted_output[-1]["word"] += word[2:]
            else:
                # This is a new word, add it to the formatted output
                # Remove 'B-' and 'I-' prefixes from the label
                label = label.replace('B-', '').replace('I-', '')
                if formatted_output and formatted_output[-1]["entity"] == label:
                    # If the current entity is the same as the previous one, append the word to the previous word
                    formatted_output[-1]["word"] += " " + word
                else:
                    # Otherwise, add a new entry to the formatted output
                    formatted_output.append({"word": word, "entity": label if label != 'O' else None})
    # for printing entire text use this below line of code
     #result = ' '.join([f"{item['word']} ({item['entity']})" if item['entity'] else item['word'] for item in formatted_output])       
        result = [f"{item['word']} ({item['entity']})" for item in formatted_output if item['entity']]
        return result









