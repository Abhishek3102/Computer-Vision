import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# 🔹 1. U-Net based GMM Model (Geometric Matching)
# -------------------------

class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, stride=1, padding=1),  # Expecting 7 channels here
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 64 * 64, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, cloth, cloth_mask, person):
        # Debugging: check input shapes
        print("Cloth shape:", cloth.shape)
        print("Cloth Mask shape:", cloth_mask.shape)
        print("Person shape:", person.shape)
        
        # Ensure cloth_mask is a 1-channel tensor
        if cloth_mask.shape[1] != 1:
            print("Error: Cloth mask should have 1 channel!")
        
        # Directly concatenate the cloth, cloth_mask (1 channel), and person (3 channels)
        x = torch.cat([cloth, cloth_mask, person], dim=1)  # Concatenate along channels
        
        # Debugging: Check the resulting shape after concatenation
        print("Concatenated input shape:", x.shape)  # Should be (batch_size, 7, H, W)
        
        # Proceed with the encoding
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, cloth.size(), align_corners=True)
        warped_cloth = F.grid_sample(cloth, grid, align_corners=True)
        
        return warped_cloth





# -------------------------
# 🔹 2. Try-On Module (TOM)
# -------------------------
class TOM(nn.Module):
    def __init__(self):
        super(TOM, self).__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, person, warped_cloth):
        x = torch.cat([person, warped_cloth], dim=1)
        output = self.unet(x)
        return output

# -------------------------
# 🔹 3. Preprocessing Function
# -------------------------
def preprocess_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def preprocess_mask(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    image = torch.tensor(image / 255.0).float().unsqueeze(0).unsqueeze(0)
    return image

# -------------------------
# 🔹 4. Load Images
# -------------------------
person_img = preprocess_image("person.jpg")  
cloth_img = preprocess_image("tshirt.png")    
cloth_mask = preprocess_mask("cloth_mask.jpg") 

print(person_img.shape)  
print(cloth_img.shape)   
print(cloth_mask.shape)  

# -------------------------
# 🔹 5. Run GMM to Warp the Cloth
# -------------------------
gmm = GMM()
warped_cloth = gmm(cloth_img, cloth_mask, person_img)

# -------------------------
# 🔹 6. Run TOM to Generate Final Try-On
# -------------------------
tom = TOM()
p_tryon = tom(person_img, warped_cloth)

# -------------------------
# 🔹 7. Save and Show Output
# -------------------------
output_image = p_tryon.squeeze().permute(1, 2, 0).detach().numpy()
plt.imshow(output_image)
plt.axis('off')
plt.savefig("tryon_result.jpg")
plt.show()
