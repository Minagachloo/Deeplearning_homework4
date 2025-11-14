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
os.makedirs("results/wgan", exist_ok=True)

# Set device for GPU/CPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration parameters
learning_rate = 0.00005
batch_size = 64
image_size = 64
noise_dimension = 128  # Size of random noise vector
num_channels = 3  # RGB images
generator_features = 64  # Base feature maps in generator
critic_features = 64  # Base feature maps in critic
num_epochs = 50
critic_iterations = 5  # Train critic 5 times per generator update
weight_clip_value = 0.01  # Weight clipping parameter for WGAN

# Prepare CIFAR10 dataset with proper transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR10 training data from your specified location
train_dataset = torchvision.datasets.CIFAR10(
    root='/home/minagachloo/Downloads/HW4/cifar-10-python',
    train=True,
    transform=transform,
    download=False  # Don't download since you have it
)

# Create data loader for batch processing
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)


class Generator(nn.Module):
    """Generator network for WGAN that creates images from noise"""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # Build generator with transposed convolutions
        self.main_network = nn.Sequential(
            # Transform noise to initial feature maps
            nn.ConvTranspose2d(noise_dimension, generator_features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 16),
            nn.ReLU(True),
            
            # Upsample and reduce channels progressively
            nn.ConvTranspose2d(generator_features * 16, generator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 8),
            nn.ReLU(True),
            
            # Continue upsampling
            nn.ConvTranspose2d(generator_features * 8, generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            
            # More upsampling
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            
            # Final layer to generate RGB image
            nn.ConvTranspose2d(generator_features * 2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, noise_input):
        """Generate fake image from random noise"""
        return self.main_network(noise_input)


class Critic(nn.Module):
    """Critic network for WGAN (not called Discriminator in WGAN)"""
    
    def __init__(self):
        super(Critic, self).__init__()
        
        # Build critic with regular convolutions
        self.main_network = nn.Sequential(
            # First layer without batch norm
            nn.Conv2d(num_channels, critic_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample and increase channels
            nn.Conv2d(critic_features, critic_features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(critic_features * 2, affine=True),  # Instance norm for WGAN
            nn.LeakyReLU(0.2, inplace=True),
            
            # Continue downsampling
            nn.Conv2d(critic_features * 2, critic_features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(critic_features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # More downsampling
            nn.Conv2d(critic_features * 4, critic_features * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(critic_features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer - no sigmoid for WGAN
            nn.Conv2d(critic_features * 8, 1, 4, 1, 0, bias=False)
        )
    
    def forward(self, image_input):
        """Score image (higher means more real)"""
        return self.main_network(image_input).view(-1)


def initialize_weights(model):
    """Initialize network weights from normal distribution"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)


# Create generator and critic networks
generator_net = Generator().to(device)
critic_net = Critic().to(device)

# Apply weight initialization
initialize_weights(generator_net)
initialize_weights(critic_net)

# RMSprop optimizers for WGAN
generator_optimizer = optim.RMSprop(generator_net.parameters(), lr=learning_rate)
critic_optimizer = optim.RMSprop(critic_net.parameters(), lr=learning_rate)

# Fixed noise for consistent visualization
fixed_noise = torch.randn(64, noise_dimension, 1, 1, device=device)

# Lists to track losses
generator_losses = []
critic_losses = []
wasserstein_distances = []
image_list = []

print("Starting WGAN Training...")

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        
        # =====================
        # Train Critic multiple times
        # =====================
        for _ in range(critic_iterations):
            critic_net.zero_grad()
            
            # Train on real images
            real_score = critic_net(real_images)
            
            # Generate and score fake images
            noise = torch.randn(current_batch_size, noise_dimension, 1, 1, device=device)
            fake_images = generator_net(noise)
            fake_score = critic_net(fake_images.detach())
            
            # Wasserstein loss for critic
            critic_loss = -torch.mean(real_score) + torch.mean(fake_score)
            critic_loss.backward()
            critic_optimizer.step()
            
            # Clip critic weights (important for WGAN)
            for param in critic_net.parameters():
                param.data.clamp_(-weight_clip_value, weight_clip_value)
        
        # =====================
        # Train Generator
        # =====================
        generator_net.zero_grad()
        
        # Generate fake images
        noise = torch.randn(current_batch_size, noise_dimension, 1, 1, device=device)
        fake_images = generator_net(noise)
        
        # Generator wants to maximize critic score for fake images
        fake_score = critic_net(fake_images)
        generator_loss = -torch.mean(fake_score)
        generator_loss.backward()
        generator_optimizer.step()
        
        # Store losses for plotting
        generator_losses.append(generator_loss.item())
        critic_losses.append(critic_loss.item())
        
        # Calculate Wasserstein distance estimate
        wasserstein_dist = torch.mean(real_score) - torch.mean(fake_score)
        wasserstein_distances.append(wasserstein_dist.item())
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"C_Loss: {critic_loss.item():.4f}, G_Loss: {generator_loss.item():.4f}, "
                  f"W_Dist: {wasserstein_dist.item():.4f}")
    
    # Save generated images each epoch
    with torch.no_grad():
        fake_images = generator_net(fixed_noise)
        vutils.save_image(fake_images, f"results/wgan/epoch_{epoch+1:03d}.png", normalize=True)
        
        # Store for final visualization
        if epoch == num_epochs - 1:
            image_list.append(vutils.make_grid(fake_images.detach().cpu(), padding=2, normalize=True))

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot losses
ax1.plot(generator_losses, label="Generator Loss", alpha=0.7)
ax1.plot(critic_losses, label="Critic Loss", alpha=0.7)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Loss")
ax1.set_title("WGAN Training Losses")
ax1.legend()

# Plot Wasserstein distance
ax2.plot(wasserstein_distances, label="Wasserstein Distance", color='green')
ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Distance")
ax2.set_title("Estimated Wasserstein Distance")
ax2.legend()

plt.tight_layout()
plt.savefig("results/wgan/training_metrics.pdf")
plt.show()

# Display and save real vs fake image comparison
print("Creating real vs fake comparison...")

# Get real images
real_batch = next(iter(train_loader))
real_images_display = real_batch[0].to(device)[:32]

# Generate fake images with best quality
with torch.no_grad():
    noise = torch.randn(32, noise_dimension, 1, 1, device=device)
    fake_images_display = generator_net(noise)

# Create side-by-side comparison
plt.figure(figsize=(15, 15))

# Real images
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real CIFAR10 Images")
real_grid = vutils.make_grid(real_images_display, padding=5, normalize=True).cpu()
plt.imshow(np.transpose(real_grid, (1, 2, 0)))

# Fake images
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("WGAN Generated Images")
fake_grid = vutils.make_grid(fake_images_display, padding=5, normalize=True).cpu()
plt.imshow(np.transpose(fake_grid, (1, 2, 0)))

plt.savefig("results/wgan/real_vs_fake_comparison.pdf")
plt.show()

# Save samples separately
vutils.save_image(real_images_display, "results/wgan/real_samples.png", normalize=True)
vutils.save_image(fake_images_display, "results/wgan/fake_samples.png", normalize=True)

# Save final models
torch.save(generator_net.state_dict(), "results/wgan/generator.pth")
torch.save(critic_net.state_dict(), "results/wgan/critic.pth")

print("WGAN training completed!")
print("Check results/wgan/ folder for generated images and comparisons")
