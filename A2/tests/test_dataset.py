import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset

from torchvision import transforms

IMAGES_ROOT = "data/lfwa/aligned_images"

transform = transforms.Compose([
    transforms.Resize((105, 105)),  # כמו Koch paper
    transforms.ToTensor()
])

pairs = parse_pairs_file(
    "data/lfwa/pairsDevTrain.txt",
    IMAGES_ROOT
)

dataset = LFWSiameseDataset(pairs, transform=transform)

print("Dataset size:", len(dataset))

img1, img2, label = dataset[0]

print("Image 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)
print("Label:", label)
