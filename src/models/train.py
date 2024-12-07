import torch
import torch.nn as nn
from episode_preprocessing import concat_episode_features
from feature_preprocessing import concat_user_features, preprocess_ms_played
from torch.utils.data import DataLoader, Dataset, random_split
from towers import SimpleTwoTower


class InteractionDataset(Dataset):
    def __init__(self, user_features, episodes_features, labels):
        self.user_features = user_features
        self.episodes_features = episodes_features
        self.labels = labels

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):
        return self.user_features[idx], self.episodes_features[idx], self.labels[idx]


# Example data
batch_size = 8
user_features = concat_user_features()
episode_features = concat_episode_features()
labels = preprocess_ms_played(log=False)

# Create the full dataset
dataset = InteractionDataset(user_features, episode_features, labels)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model setup
user_embed_dim = user_features.shape[1]
item_embed_dim = episode_features.shape[1]
model = SimpleTwoTower(user_embed_dim, item_embed_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    all_outputs = []
    all_labels = []

    # Training phase
    for user_features_batch, episode_features_batch, labels_batch in train_dataloader:
        outputs = model(user_features_batch, episode_features_batch).squeeze(dim=-1)

        loss = criterion(outputs, labels_batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store outputs and labels for MAE/RMSE computation
        all_outputs.append(outputs.detach())
        all_labels.append(labels_batch.detach())

    # Combine outputs and labels from all batches
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    # Compute MAE and RMSE for the training phase
    train_mae = torch.abs(all_outputs - all_labels).mean()
    train_rmse = torch.sqrt(((all_outputs - all_labels) ** 2).mean())

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_outputs = []
    val_labels = []

    with torch.no_grad():  # No gradients needed during evaluation
        for user_features_batch, episode_features_batch, labels_batch in val_dataloader:
            outputs = model(user_features_batch, episode_features_batch).squeeze(dim=-1)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            # Store outputs and labels for MAE/RMSE computation
            val_outputs.append(outputs.detach())
            val_labels.append(labels_batch.detach())

        # Combine outputs and labels from all batches
        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels)

        # Compute MAE and RMSE for the validation phase
        val_mae = torch.abs(val_outputs - val_labels).mean()
        val_rmse = torch.sqrt(((val_outputs - val_labels) ** 2).mean())

    # Print metrics for each epoch
    print(
        f"Epoch {epoch + 1}, "
        f"Train Loss: {epoch_loss/len(train_dataloader):.4f}, "
        f"Train MAE: {train_mae.item():.4f}, Train RMSE: {train_rmse.item():.4f}, "
        f"Val Loss: {val_loss/len(val_dataloader):.4f}, "
        f"Val MAE: {val_mae.item():.4f}, Val RMSE: {val_rmse.item():.4f}"
    )
