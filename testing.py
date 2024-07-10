import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from train import config, UNet, weights_init  # Assuming config, UNet, and weights_init are imported from train.py

# Constants
TEST_DATASET_HAZY_PATH = './val/hazy'
TEST_DATASET_OUTPUT_PATH = './dehazed_output2'

# Ensure device availability
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

# Define the custom validation dataset class
class CustomValDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, filename

# Transformation to apply to images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Create dataset and dataloader
val_dataset = CustomValDataset(TEST_DATASET_HAZY_PATH, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load pretrained model
loaded_state = torch.load("./pix2pix_140970.pth", map_location=device)
gen = UNet(config['input_dim'], config['output_dim']).to(device)
gen.apply(weights_init)
gen.load_state_dict(loaded_state["gen"])

# Iterate through the validation dataset
for image, filename in val_dataloader:
    try:
        # Resize image to target shape
        condition = nn.functional.interpolate(image, size=config['target_shape'], mode='bicubic')
        condition = condition.to(device)

        # Generate dehazed image
        with torch.no_grad():
            fake = gen(condition)

        # Convert generated image to PIL format for saving
        fake_image = fake.cpu().squeeze(0)
        fake_image = transforms.ToPILImage()(fake_image)

        # Save the image
        save_path = os.path.join(TEST_DATASET_OUTPUT_PATH, filename[0])
        fake_image.save(save_path)
        print(f"Saved dehazed image: {save_path}")

    except Exception as e:
        print(f"Error processing {filename[0]}: {e}")
