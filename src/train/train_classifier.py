import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextCategoryModel(nn.Module):
    def __init__(self, num_categories, hidden_dim=128, text_embedding_dim=384):
        super(TextCategoryModel, self).__init__()
        self.text_embedding_dim = text_embedding_dim  # From all-MiniLM-L6-v2
        self.fc_text = nn.Linear(self.text_embedding_dim, hidden_dim)  # From 384 to hidden_dim
        self.fc_combined = nn.Linear(hidden_dim, hidden_dim)  # Keep the same dimension for next layer
        self.output_layer = nn.Linear(hidden_dim, num_categories)  # Output layer for multi-class

    def forward(self, text_embedding):
        # Pass through text embedding layer
        text_out = F.relu(self.fc_text(text_embedding))
        combined_out = F.relu(self.fc_combined(text_out))  # Apply ReLU activation after combining
        output = self.output_layer(combined_out)  # Logits for multi-class
        return output


def get_channel_function(df, set, category):
    rows = []
    if set == 'ACTION':
        prev = 'THEN '
    elif set == 'TRIGGER':
        prev = 'IF '
    else:
        prev = ''

    for index, row in tqdm(df.iterrows(), total=len(df)):
        cleaned_text = row['output'].replace(f"{set} SERVICE:", "").strip()
        parts = cleaned_text.split(f", {set} EVENT:")

        if len(parts) == 2 and parts[0] and parts[1]:
            rows.append({
                "input": prev + parts[0].strip() + parts[1].strip(),
                "channel": parts[0].strip(),
                "function": parts[1].strip()
            })

    return pd.DataFrame(rows, columns=["input", "channel", f"function"]).merge(category, left_on='channel',
                                                                               right_on='service', how='left')

name = 'NAME'
set = 'action' #trigger or action
target = 'function' #channel or function or both

if target == 'default':
    sentence_model = SentenceTransformer(f'{name}')
else:
    sentence_model = SentenceTransformer(f'../../models/{name}/{set}_{target}_{name}_embedder')

service_pairing = pd.read_csv(f"../../data/services/{set}_services.csv")

train_dataset = pd.read_csv(f"../../data/dataset/train_{set}.csv")
gold_dataset = pd.read_csv(f"../../data/dataset/gold/test_{set}.csv")
noisy_dataset = pd.read_csv(f"../../data/dataset/noisy/test_{set}.csv")

train_dataset = get_channel_function(train_dataset, set.upper(), service_pairing)
gold_dataset = get_channel_function(gold_dataset, set.upper(), service_pairing)
noisy_dataset = get_channel_function(noisy_dataset, set.upper(), service_pairing)


def get_torch_dataset(df, sentence_model, num_categories):
    embeddings = sentence_model.encode(df['input'].tolist(), convert_to_tensor=True, device=device).cpu()
    labels = torch.zeros(len(df), num_categories)
    for i, row in df.iterrows():
        labels[i][int(row['category']) - 1] = 1

    return torch.utils.data.TensorDataset(embeddings, labels)


num_categories = 14
train_tensor_dataset = get_torch_dataset(train_dataset, sentence_model, num_categories)
gold_tensor_dataset = get_torch_dataset(gold_dataset, sentence_model, num_categories)
noisy_tensor_dataset = get_torch_dataset(noisy_dataset, sentence_model, num_categories)

batch_size = 128
num_epochs = 20
lr = 0.001

category_classifier = TextCategoryModel(num_categories,
                                        text_embedding_dim=sentence_model.get_sentence_embedding_dimension()).to(device)
train_loader = torch.utils.data.DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
gold_loader = torch.utils.data.DataLoader(gold_tensor_dataset, batch_size=batch_size, shuffle=False)
noisy_loader = torch.utils.data.DataLoader(noisy_tensor_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use CrossEntropyLoss for multi-class
optimizer = torch.optim.Adam(category_classifier.parameters(), lr=lr)

for epoch in range(num_epochs):
    category_classifier.train()  # Set the model to training mode
    total_loss = 0

    for batch_X, batch_y in train_loader:
        # Move batch_X and batch_y to the device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = category_classifier(batch_X)

        # Compute loss
        loss = criterion(outputs, batch_y)  # Compute loss directly with batch_y as target
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')

os.makedirs(f'../../models/{name}/{set}_{target}_classifier/', exist_ok=True)
torch.save(category_classifier.state_dict(), f'../../models/{name}/{set}_{target}_classifier/{set}_classifier.pth')

categories_dict = {
    'Smart Devices': 0,
    'Wearable devices': 1,
    'Time & location': 2,
    'Online services': 3,
    'Personal data managers and schedulers': 4,
    'Iot Hubs/Integration solutions': 5,
    'Automobiles': 6,
    'Social networking, blogging, sharing platforms': 7,
    'Cloud storage': 8,
    'Messaging, team collaboration, VoIP': 9,
    'Smartphone applications': 10,
    'RSS and recommendation systems': 11,
    'Mail services': 12,
    'Other': 13,
}

with torch.no_grad():
    total_correct = 0
    total_samples = 0

    all_predicted = []
    all_true = []

    for batch_X, batch_y in gold_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = category_classifier(batch_X)

        probabilities = torch.sigmoid(outputs)

        true_classes = batch_y.argmax(dim=1)
        predicted_classes = probabilities.argmax(dim=1)

        total_correct += (predicted_classes == true_classes).sum().item()
        total_samples += batch_y.size(0)

        all_predicted.extend(predicted_classes.cpu().numpy())
        all_true.extend(true_classes.cpu().numpy())

    if total_samples > 0:
        accuracy = total_correct / total_samples
    else:
        accuracy = 0.0

    print(f'Test Accuracy: {accuracy * 100:.2f}%')

all_true_array = np.array(all_true)
all_predicted_array = np.array(all_predicted)

# Compute confusion matrix (no argmax needed as we already have class indices)
cm = confusion_matrix(all_true_array, all_predicted_array)
row_sums = cm.sum(axis=1, keepdims=True)  # Sum each row
cm_normalized = cm / row_sums
cm_normalized = np.log(cm_normalized + 1e-9)

# Plot confusion matrix
plt.figure(figsize=(10, 8))

heatmap = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                      xticklabels=[value + 1 for value in categories_dict.values()],
                      yticklabels=[value + 1 for value in categories_dict.values()])

colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Test GOLD', fontsize=20, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
plt.ylabel('True Label', fontsize=18, fontweight='bold')
plt.savefig(f'../../results/{set}_{target}_{name}_classifier_gold_confusion_matrix.svg', format="svg")
plt.show()

with torch.no_grad():
    total_correct = 0
    total_samples = 0

    all_predicted = []
    all_true = []

    for batch_X, batch_y in noisy_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = category_classifier(batch_X)

        probabilities = torch.sigmoid(outputs)

        true_classes = batch_y.argmax(dim=1)
        predicted_classes = probabilities.argmax(dim=1)

        total_correct += (predicted_classes == true_classes).sum().item()
        total_samples += batch_y.size(0)

        all_predicted.extend(predicted_classes.cpu().numpy())
        all_true.extend(true_classes.cpu().numpy())

    if total_samples > 0:
        accuracy = total_correct / total_samples
    else:
        accuracy = 0.0

    print(f'Test Accuracy: {accuracy * 100:.2f}%')

all_true_array = np.array(all_true)
all_predicted_array = np.array(all_predicted)

# Compute confusion matrix (no argmax needed as we already have class indices)
cm = confusion_matrix(all_true_array, all_predicted_array)
row_sums = cm.sum(axis=1, keepdims=True)  # Sum each row
cm_normalized = cm / row_sums
cm_normalized = np.log(cm_normalized + 1e-9)

# Plot confusion matrix
plt.figure(figsize=(10, 8))

heatmap = sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                      xticklabels=[value + 1 for value in categories_dict.values()],
                      yticklabels=[value + 1 for value in categories_dict.values()])

colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.title('Test NOISY', fontsize=20, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
plt.ylabel('True Label', fontsize=18, fontweight='bold')
plt.savefig(f'../../results/{set}_{target}_{name}_classifier_noisy_confusion_matrix.svg', format="svg")
plt.show()
