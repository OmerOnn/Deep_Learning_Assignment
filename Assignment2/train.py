import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datasets.lfw_pairs import LFWDataset

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

dataset = LFWDataset(
    pairs_file="data/pairsDevTrain.txt",
    lfw_root="data/lfw2", 
    transform=transform
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for img1, img2, label in loader:
    print(img1.shape, img2.shape, label)
    break