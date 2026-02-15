# !pip install diffusers transformers accelerate torch torchvision matplotlib
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import glob
import os
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler
import torch.optim as optim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import shutil
from torch import nn
from torchsummary import summary

os.chdir('/my/input/dataset')
destination_folder = '/my/destination/folder'
os.makedirs(destination_folder, exist_ok=True) # Create destination if it doesn't exist

for filename in os.listdir():
    if os.path.isfile(filename):
        shutil.copy(filename, destination_folder)
    else:
        print(f"Skipping {filename}")


# Create UNet model
class ModifiedUNet2DModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define extra_conv (Optional)
        self.extra_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, sample, timesteps=1, **kwargs):
        output = super().forward(sample, timesteps, **kwargs)
        output.sample = self.extra_conv(output.sample)
        return output

# Instantiate the model with chosen hyperparameters
model = ModifiedUNet2DModel(
    sample_size=128,
    in_channels=1,
    out_channels=1,
    layers_per_block=3,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
)

# Run on GPU if available (for quicker training)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


# Create Schedulers

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

# Optional, adjust learning rate if loss is not performing well
scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    patience=5,
                                                    factor=0.8,
                                                    cooldown=1,
                                                    verbose=True)


# Directory to save checkpoints
# Optional, but reccommended for longer training times
checkpoint_dir = '/my/checkpoint/directory'
os.makedirs(checkpoint_dir,exist_ok=True)

# Saves each checkpoint by their epoch
# Optional --If best is true, it writes "BEST" next to the name, for tracking best performance
def save_checkpoint(model, optimizer, epoch, best=False):
    """Save the model and optimizer state dictionaries along with the epoch."""
    filename = f'checkpoint_epoch_{epoch}{"_BEST" if best else ""}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Optionally, save the scheduler state
        'scheduler_lr_state_dict': scheduler_lr.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Load checkpoint (Only if model has been trained previously)
def load_checkpoint(model, optimizer, checkpoint_path):
    """Load the model and optimizer state dictionaries from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch.
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch

# Load latest checkpoint if available (change checkpoint directory name)
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    start_epoch = load_checkpoint(model, optimizer, latest_checkpoint)
else:
    start_epoch = 0
    print("Starting from scratch")

num_epochs = 300  # Adjust based on available resources

class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d != ".ipynb_checkpoints"]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Convert to [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Maps [0,1] to [-1,1]
])

dataset = CustomImageFolder(
    root="/my/images/folder/",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Optional -- create early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=10, min_radius=0):
        self.patience = patience
        self.min_radius = min_radius
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = None  # Track the best epoch

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch  # Save the best epoch
            return False, True  # Continue training, best checkpoint should be saved
        elif val_loss < self.best_loss + self.min_radius:
            self.counter = 0
            return False,False
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered!")
            return True, False  # Stop training, but no new best checkpoint
        return False, False  # Continue training, no new best checkpoint

# You can adjust patience accordingly (If the loss does not improve after patience epochs, it will break)
early_stopping = EarlyStopping(patience=50,min_radius = 0.001)

