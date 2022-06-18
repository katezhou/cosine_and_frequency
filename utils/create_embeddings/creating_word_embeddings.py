#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
import numpy as np
import math
import csv
import tqdm
import re

from transformers import pipeline
from transformers import BertConfig
from transformers import BertModel

import random
from transformers import *

torch.cuda.is_available()

class bert_model():
    def __init__(self, name):
        self.name = name
        self.config = BertConfig(output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(name, max_length=512, truncation=True)
        # Whether the model returns all hidden-states.
        self.object = BertModel.from_pretrained(name,
                                  output_hidden_states = True)

def get_id(keyword, tokenizer):    
    tokenized_text = tokenizer.tokenize("[CLS] " + keyword + " [SEP]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens[1:-1]

def average_last_4(token_embeddings):
    token_vecs_cat = []

    for token in token_embeddings:
        #stack all into 2d array
        cat_vec = torch.stack((token[-1], token[-2], token[-3], token[-4]), dim=0)
        #take the average across columns
        cat_vec = torch.mean(cat_vec, 0)
        token_vecs_cat.append(cat_vec)
    return token_vecs_cat


def extract_word_embeddings_average_first_token(model, text, ids):    
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = model.tokenizer.tokenize(marked_text)
    indexed_tokens = model.tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.object.eval()

    with torch.no_grad():
        outputs = model.object(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    
    #WORD EMBEDDING
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_average_last_4 = average_last_4(token_embeddings)

    #find where keyword is
    word_embedding = []
    try:
        #find the sequence
        #take the embedding of the first item in the sequence
        index = [(i, i+len(ids)) for i in range(len(indexed_tokens)) if indexed_tokens[i:i+len(ids)] == ids][0][0]
        word_embedding = token_vecs_average_last_4[index]

    except:
        word_embedding = None
        #this happens when e.g., the example contains Albanian instead of just Albania --- we'll want to skip this example
        print("Skip sentence")
        return None, None 
    
    #SENTENCE EMBEDDING
    token_vecs = hidden_states[-2][0]

    #Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return word_embedding, sentence_embedding 


def extract_word_embeddings_average_tokens(model, text, ids):
    
    indexed_tokens = model.tokenizer(text, truncation= True, max_length = 512)['input_ids']
    
    segments_ids = [1] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.object.eval()

    with torch.no_grad():
        outputs = model.object(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    
    #WORD EMBEDDING
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs_average_last_4 = average_last_4(token_embeddings)

    #find where keyword is
    try:
        index = [(i, i+len(ids)) for i in range(len(indexed_tokens)) if indexed_tokens[i:i+len(ids)] == ids][0][0]
    except:
        #this happens when e.g., the example contains Albanian instead of just Albania --- we'll want to skip this example
        print("Skip sentence", text)
        return None, None 
    
    word_embedding = []
    for x in range(index, index+len(ids)):
        #concat all subtokens
        word_embedding.append(token_vecs_average_last_4[x])
    
    #stack them
    word_embedding = torch.stack(word_embedding, dim=0)
    #take the average of subtokens
    word_embedding = torch.mean(word_embedding, 0)
        
    #SENTENCE EMBEDDING
    token_vecs = hidden_states[-2][0]

    #Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return word_embedding, sentence_embedding 


def run_analysis(model, data, indices, ids):
    word_embeddings = []
    sentence_embeddings = []
    index_numbers = []
    for n, row in data.iterrows():
        word_embedding, sentence_embedding = extract_word_embeddings_average_tokens(model, row[0], ids)
        if (word_embedding is not None):
            word_embeddings.append(word_embedding)
            sentence_embeddings.append(sentence_embedding)
            index_numbers.append(indices[n])
            #if you found one, move on
            #break

    return word_embeddings, sentence_embeddings, index_numbers

bert_base_cased = bert_model('bert-base-cased')

# # General Method of Creating Embeddings
def create_embeddings(filename, data, model=bert_base_cased):
    append_write = 'a' 

    with open(filename, append_write, newline='') as f:
        writer = csv.writer(f, delimiter=',')
        #change to keyword tag at times
        for name, group in data.groupby(['keyword']):
            #when group is not the same as keyword
            name_match = group['keyword'].values[0]
            ids = get_id(name_match, model.tokenizer)
            word_embeddings, sentence_embeddings, row_numbers = run_analysis(model, pd.DataFrame(group['sentence'].values), group['index'].values, ids)
            for word_embedding, sentence_embedding, row_number in zip(word_embeddings, sentence_embeddings, row_numbers):
                writer.writerow(word_embedding.tolist() + sentence_embedding.tolist() + [row_number] + [name] + [len(ids)])
            print(name)

