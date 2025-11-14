import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# Create folders to save results
os.makedirs("results/dcgan", exist_ok=True)

# Set device for GPU/CPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration parameters
learning_rate = 0.0002
batch_size = 128
image_size = 64
noise_dimension = 100  # Size of random noise vector
num_channels = 3  # RGB images
generator_features = 64  # Base feature maps in generator
discriminator_features = 64  # Base feature maps in discriminator
num_epochs = 50
beta1 = 0.5  # Adam optimizer parameter

# Prepare CIFAR10 dataset with proper transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR10 training data
train_dataset = torchvision.datasets.CIFAR10(
    root='/home/minagachloo/Downloads/HW4/cifar-10-python',
    train=True,
    transform=transform,
    download=False
)

# Create data loader for batch processing
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)


class Generator(nn.Module):
    """Generator network that creates fake images from random noise"""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # Build the generator architecture using transposed convolutions
        self.main_network = nn.Sequential(
            # Transform noise vector into feature maps
            nn.ConvTranspose2d(noise_dimension, generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 8),
            nn.ReLU(True),
            
            # Progressively upsample to increase spatial dimensions
            nn.ConvTranspose2d(generator_features * 8, generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            
            # Continue upsampling and reducing channels
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            
            # Further upsampling
            nn.ConvTranspose2d(generator_features * 2, generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features),
            nn.ReLU(True),
            
            # Final layer outputs RGB image
            nn.ConvTranspose2d(generator_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, noise_input):
        """Generate fake image from noise input"""
        return self.main_network(noise_input)


class Discriminator(nn.Module):
    """Discriminator network that classifies images as real or fake"""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Build the discriminator architecture using regular convolutions
        self.main_network = nn.Sequential(
            # Initial convolution without batch normalization
            nn.Conv2d(num_channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Progressively downsample and increase channels
            nn.Conv2d(discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Continue downsampling
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Further downsampling
            nn.Conv2d(discriminator_features * 4, discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer outputs single probability value
            nn.Conv2d(discriminator_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, image_input):
        """Classify image as real or fake"""
        return self.main_network(image_input).view(-1, 1).squeeze(1)


def initialize_weights(model):
    """Initialize network weights from normal distribution"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)


# Create generator and discriminator networks
generator_net = Generator().to(device)
discriminator_net = Discriminator().to(device)

# Apply custom weight initialization
initialize_weights(generator_net)
initialize_weights(discriminator_net)

# Binary cross entropy loss for adversarial training
criterion = nn.BCELoss()

# Adam optimizers for both networks
generator_optimizer = optim.Adam(generator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Fixed noise for visualization during training
fixed_noise = torch.randn(64, noise_dimension, 1, 1, device=device)

# Lists to store training losses
generator_losses = []
discriminator_losses = []
image_list = []  # Store generated images for visualization

print("Starting DCGAN Training...")

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        
        # Labels for real and fake images
        real_label = torch.ones(current_batch_size, device=device)
        fake_label = torch.zeros(current_batch_size, device=device)
        
        # ====================
        # Train Discriminator
        # ====================
        discriminator_net.zero_grad()
        
        # Train on real images
        real_output = discriminator_net(real_images)
        discriminator_loss_real = criterion(real_output, real_label)
        discriminator_loss_real.backward()
        
        # Generate fake images and train on them
        noise = torch.randn(current_batch_size, noise_dimension, 1, 1, device=device)
        fake_images = generator_net(noise)
        fake_output = discriminator_net(fake_images.detach())
        discriminator_loss_fake = criterion(fake_output, fake_label)
        discriminator_loss_fake.backward()
        
        # Update discriminator weights
        discriminator_optimizer.step()
        
        # ====================
        # Train Generator
        # ====================
        generator_net.zero_grad()
        
        # Generator tries to fool discriminator
        fake_output = discriminator_net(fake_images)
        generator_loss = criterion(fake_output, real_label)
        generator_loss.backward()
        
        # Update generator weights
        generator_optimizer.step()
        
        # Store losses for plotting
        generator_losses.append(generator_loss.item())
        discriminator_losses.append(discriminator_loss_real.item() + discriminator_loss_fake.item())
        
        # Print training progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"D_Loss: {discriminator_losses[-1]:.4f}, G_Loss: {generator_loss.item():.4f}")
    
    # Save generated images each epoch
    with torch.no_grad():
        fake_images = generator_net(fixed_noise)
        vutils.save_image(fake_images, f"results/dcgan/epoch_{epoch+1:03d}.png", normalize=True)
        
        # Store for final comparison
        if epoch == num_epochs - 1:
            image_list.append(vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True))

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("DCGAN Training Loss")
plt.legend()
plt.savefig("results/dcgan/training_loss.pdf")
plt.show()

# Display and save real vs fake image comparison
print("Creating real vs fake comparison...")

# Get a batch of real images
real_batch = next(iter(train_loader))
real_images_display = real_batch[0].to(device)[:32]  # Take first 32 images

# Generate fake images
with torch.no_grad():
    noise = torch.randn(32, noise_dimension, 1, 1, device=device)
    fake_images_display = generator_net(noise)

# Create comparison figure
plt.figure(figsize=(15, 15))

# Display real images
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real CIFAR10 Images")
real_grid = vutils.make_grid(real_images_display[:32], padding=5, normalize=True).cpu()
plt.imshow(np.transpose(real_grid, (1, 2, 0)))

# Display fake images
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Generated Fake Images")
fake_grid = vutils.make_grid(fake_images_display, padding=5, normalize=True).cpu()
plt.imshow(np.transpose(fake_grid, (1, 2, 0)))

plt.savefig("results/dcgan/real_vs_fake_comparison.pdf")
plt.show()

# Save individual real and fake samples
vutils.save_image(real_images_display, "results/dcgan/real_samples.png", normalize=True)
vutils.save_image(fake_images_display, "results/dcgan/fake_samples.png", normalize=True)

# Save final models
torch.save(generator_net.state_dict(), "results/dcgan/generator.pth")
torch.save(discriminator_net.state_dict(), "results/dcgan/discriminator.pth")

print("DCGAN training completed!")
print("Check results/dcgan/ folder for generated images and comparisons")
