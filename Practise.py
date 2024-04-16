import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Load Dataset
true_data = pd.read_csv('gossipcop_real.csv')
fake_data = pd.read_csv('gossipcop_fake.csv')

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    text = ''.join([char for char in text if char.isalnum() or char in [' ', "'"]])
    return text

true_data['title'] = true_data['title'].apply(preprocess_text)
fake_data['title'] = fake_data['title'].apply(preprocess_text)

# Generate labels True/Fake under new Target Column in 'true_data' and 'fake_data'
true_data['Target'] = ['True'] * len(true_data)
fake_data['Target'] = ['Fake'] * len(fake_data)

# Merge 'true_data' and 'fake_data', by random mixing into a single df called 'data'
fake_news_data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text and pad/truncate sequences
max_length = 128  # Maximum sequence length
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True) for text in fake_news_data['title']]

# Pad sequences
input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(tokenized_text) for tokenized_text in tokenized_texts], batch_first=True)

# Create attention masks
attention_masks = [[int(token_id > 0) for token_id in input_id] for input_id in input_ids]

# Prepare Labels
labels = [1 if label == 'Fake' else 0 for label in fake_news_data['Target']]

# Convert data to PyTorch tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

# Split data into training and testing sets
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(input_ids, attention_masks, labels, test_size=0.2, random_state=42)

# Define BERT model for sequence classification with fewer layers
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  
    output_attentions=False,
    output_hidden_states=False,
    num_hidden_layers=6 
)

# Fine-tuning BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_inputs) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Data loaders
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, masks, labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(input_ids, token_type_ids=None, attention_mask=masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid exploding gradients
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for batch in test_dataloader:
        input_ids, masks, labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=masks)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat)
    recall = recall_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat)
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Training loss: {avg_train_loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print("\n")
