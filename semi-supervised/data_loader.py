
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def load_and_preprocess_images(path, image_size=(64, 64), batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),  # Adding a center crop
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

