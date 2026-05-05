import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.lfw_pairs import LFWDataset
from models.siamese_koch import SiameseKoch


# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Data
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

dataset = LFWDataset(
    pairs_file="data/pairsDevTrain.txt",
    lfw_root="data/lfw2",
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = SiameseKoch().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# -----------------------------
# Training loop
# -----------------------------
model.train()

print("Number of samples:", len(dataset))
print("Number of batches:", len(loader))

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for i, (img1, img2, label) in enumerate(loader):
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)

        label = label.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        output = model(img1, img2)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # Print every 100 batches
        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{i}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")