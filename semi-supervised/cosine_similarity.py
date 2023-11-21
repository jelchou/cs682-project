import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def image_cosine_similarity(image1_path, image2_path, image_size):

  transform = transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
  ])
  
  image1 = transform(Image.open(image1_path)).unsqueeze(0)
  image2 = transform(Image.open(image2_path)).unsqueeze(0)
  
  image1_tensor = torch.autograd.Variable(image1, requires_grad=False)
  image2_tensor = torch.autograd.Variable(image2, requires_grad=False)
  
  image1_flat = image1_tensor.view(-1)
  image2_flat = image2_tensor.view(-1)
  
  cosine_similarity = F.cosine_similarity(image1_flat, image2_flat, dim=0)
  
  print(f'Cosine Similarity: {cosine_similarity.item()}')
  return cosine_similarity.item()
