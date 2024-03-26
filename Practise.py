#Practise

#Setup Environment

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import lightgbm as lgb
import numpy as np
import pylance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#Load BERT

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load Dataset

true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Generate labels for True/Fake news
true_data['Target'] = 1
fake_data['Target'] = 0

# Merge and shuffle data
data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)

#Tokenization of text using BERT tokenizer

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

#Splitting Data into train and test set 

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)

# Define BERT Model Architecture
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)  # Output size 1 for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Output of [CLS] token
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output
        
# Initialize BERT model, optimizer, and loss function

model = BERTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Train BERT Model

batch_size = 32
epochs = 3

for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_inputs), batch_size):
        optimizer.zero_grad()
        batch_inputs = train_inputs[i:i+batch_size]
        batch_masks = train_masks[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        outputs = model(batch_inputs, batch_masks)
        loss = criterion(outputs.squeeze(), batch_labels.float())
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate BERT Model

model.eval()
with torch.no_grad():
    outputs = model(test_inputs, test_masks)
    predictions = torch.round(torch.sigmoid(outputs)).squeeze()
    accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
    print(f'Accuracy: {accuracy}')
    
# Extract BERT embeddings for entire dataset

model.eval()
with torch.no_grad():
    all_outputs = model(input_ids, attention_masks).squeeze().numpy()

# Train LightGBM classifier

X_train, X_test, y_train, y_test = train_test_split(all_outputs, labels.numpy(), test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10, verbose_eval=10)

# Evaluate LightGBM classifier

y_pred = np.round(bst.predict(X_test, num_iteration=bst.best_iteration))
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {accuracy}")










