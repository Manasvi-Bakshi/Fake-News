#Practise

#Setup Environment

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#Load BERT

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

#Sample data

true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')
true_data['Target'] = ['True']*len(true_data)
fake_data['Target'] = ['Fake']*len(fake_data)
data = true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index'])

#Tokenization BERT

max_length = 128
input_ids = []
attention_masks = []
for text in data:
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True,max_length=max_length, padding='max_length', truncation=True,return_attention_mask=True,return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks= torch.cat(attention_masks, dim=0)
with torch.no_grad():
    outputs= bert_model(input_ids, attention_mask=attention_masks)
bert_embeddings = outputs[0][:,0,:].numpy()

#Splitting Data

x_train, x_test, y_train, y_test = train_test_split(bert_embeddings, labels, test_size=0.2, random_state=42) 

#Train LightGBM
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)









