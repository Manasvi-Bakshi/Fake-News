#Practise

#Setup Environment

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import lightgbm as lgb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


#Load BERT

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load Dataset

true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Generate labels for True/Fake news

true_data['Target'] = ['True']*len(true_data)
fake_data['Target'] = ['Fake']*len(fake_data)

# Merge and shuffle data

fake_news_data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)

# Split the dataset into training and testing sets

train_data, test_data = train_test_split(fake_news_data, test_size=0.2, random_state=42)

# Define BERT tokenizer and model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

#Tokenization of text using BERT tokenizer

def tokenize_data(data):
    tokenized_data = tokenizer(data['title'].tolist(), padding=True, truncation=True, return_tensors="pt")
    tokenized_data['labels'] = torch.tensor([0 if label == 'True' else 1 for label in data['Target'].tolist()])
    return tokenized_data

train_tokenized = tokenize_data(train_data)
test_tokenized = tokenize_data(test_data)


# Custom Dataset class

class NewsDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Define data loaders

train_dataset = NewsDataset(train_tokenized)
test_dataset = NewsDataset(test_tokenized)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    output_dir="./results",
    overwrite_output_dir=True,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda pred: {
        "keys": pred.keys() if isinstance(pred, dict) else None,
        "predictions_shape": pred.predictions.shape if hasattr(pred, 'predictions') else None,
        "label_ids_shape": pred.label_ids.shape if hasattr(pred, 'label_ids') else None
    }
)

train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
eval_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)

# Fine-tune BERT

for step, batch in enumerate(train_dataloader):
    inputs = {key: value.to(trainer.args.device) for key, value in batch.items()}
    print(f"Step {step}, Inputs: {inputs.keys()}")  # Print inputs keys
    trainer.train()
# Train LightGBM classifier
lgbm_classifier = LGBMClassifier()
lgbm_classifier.fit(X_train, y_train)

# Predict labels on the test set

y_pred = lgbm_classifier.predict(X_test)

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Fake')
recall = recall_score(y_test, y_pred, pos_label='Fake')
f1 = f1_score(y_test, y_pred, pos_label='Fake')
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision (Fake):", precision)
print("Recall (Fake):", recall)
print("F1-score (Fake):", f1)
print("Classification Report:\n", classification_rep)










