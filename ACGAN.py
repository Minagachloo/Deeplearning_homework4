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
os.makedirs("results/acgan", exist_ok=True)

# Set device for GPU/CPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration parameters
learning_rate = 0.00005
batch_size = 64
image_size = 64
noise_dimension = 100  # Size of random noise vector
num_classes = 10  # CIFAR10 has 10 classes
embedding_dimension = 100  # Size of class embedding
num_channels = 3  # RGB images
generator_features = 64  # Base feature maps in generator
discriminator_features = 64  # Base feature maps in discriminator
num_epochs = 40

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
    """ACGAN Generator that creates images conditioned on class labels"""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # Embedding layer for class labels
        self.label_embedding = nn.Embedding(num_classes, embedding_dimension)
        
        # Generator architecture with label conditioning
        self.main_network = nn.Sequential(
            # Combine noise and label embedding
            nn.ConvTranspose2d(noise_dimension + embedding_dimension, generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_features * 8),
            nn.ReLU(True),
            
            # Progressively upsample
            nn.ConvTranspose2d(generator_features * 8, generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 4),
            nn.ReLU(True),
            
            # Continue upsampling
            nn.ConvTranspose2d(generator_features * 4, generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features * 2),
            nn.ReLU(True),
            
            # More upsampling
            nn.ConvTranspose2d(generator_features * 2, generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_features),
            nn.ReLU(True),
            
            # Final layer generates RGB image
            nn.ConvTranspose2d(generator_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, noise_input, class_labels):
        """Generate images conditioned on class labels"""
        # Embed class labels and reshape for concatenation
        embedded_labels = self.label_embedding(class_labels)
        embedded_labels = embedded_labels.unsqueeze(2).unsqueeze(3)
        
        # Concatenate noise and label embedding
        combined_input = torch.cat([noise_input, embedded_labels], dim=1)
        
        # Generate image
        return self.main_network(combined_input)


class Discriminator(nn.Module):
    """ACGAN Discriminator that outputs both real/fake and class predictions"""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Shared feature extraction layers
        self.feature_extractor = nn.Sequential(
            # First convolution layer
            nn.Conv2d(num_channels, discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample and increase channels
            nn.Conv2d(discriminator_features, discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Continue downsampling
            nn.Conv2d(discriminator_features * 2, discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final feature extraction
            nn.Conv2d(discriminator_features * 4, discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output layer for real/fake classification
        self.adversarial_output = nn.Sequential(
            nn.Conv2d(discriminator_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Output layer for class classification
        self.auxiliary_output = nn.Sequential(
            nn.Conv2d(discriminator_features * 8, num_classes, 4, 1, 0, bias=False)
        )
    
    def forward(self, image_input):
        """Output both validity (real/fake) and class prediction"""
        # Extract features
        features = self.feature_extractor(image_input)
        
        # Get validity prediction (real or fake)
        validity = self.adversarial_output(features).view(-1)
        
        # Get class prediction
        class_prediction = self.auxiliary_output(features).view(-1, num_classes)
        
        return validity, class_prediction


def initialize_weights(model):
    """Initialize network weights from normal distribution"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)


# Create generator and discriminator networks
generator_net = Generator().to(device)
discriminator_net = Discriminator().to(device)

# Apply weight initialization
initialize_weights(generator_net)
initialize_weights(discriminator_net)

# Loss functions for ACGAN
adversarial_loss = nn.BCELoss()  # For real/fake classification
auxiliary_loss = nn.CrossEntropyLoss()  # For class classification

# Adam optimizers
generator_optimizer = optim.Adam(generator_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Fixed noise and labels for visualization
num_samples = 100  # 10x10 grid
fixed_noise = torch.randn(num_samples, noise_dimension, 1, 1, device=device)
fixed_labels = torch.tensor([i % 10 for i in range(num_samples)], device=device)  # One of each class

# Lists to store training metrics
generator_losses = []
discriminator_losses = []
accuracies = []

print("Starting ACGAN Training...")
print("Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

# Training loop
for epoch in range(num_epochs):
    running_accuracy = 0.0
    num_batches = 0
    
    for batch_idx, (real_images, real_labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        current_batch_size = real_images.size(0)
        
        # Labels for adversarial loss
        valid_labels = torch.ones(current_batch_size, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        discriminator_optimizer.zero_grad()
        
        # Train on real images
        real_validity, real_class_pred = discriminator_net(real_images)
        real_adv_loss = adversarial_loss(real_validity, valid_labels)
        real_aux_loss = auxiliary_loss(real_class_pred, real_labels)
        real_loss = real_adv_loss + real_aux_loss
        
        # Train on fake images
        noise = torch.randn(current_batch_size, noise_dimension, 1, 1, device=device)
        generated_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        fake_images = generator_net(noise, generated_labels)
        
        fake_validity, fake_class_pred = discriminator_net(fake_images.detach())
        fake_adv_loss = adversarial_loss(fake_validity, fake_labels)
        fake_aux_loss = auxiliary_loss(fake_class_pred, generated_labels)
        fake_loss = fake_adv_loss + fake_aux_loss
        
        # Total discriminator loss
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Calculate accuracy for real images
        predicted_classes = torch.argmax(real_class_pred, dim=1)
        accuracy = (predicted_classes == real_labels).float().mean()
        running_accuracy += accuracy.item()
        num_batches += 1
        

        generator_optimizer.zero_grad()
        
        # Generator wants discriminator to think fake images are real
        fake_validity, fake_class_pred = discriminator_net(fake_images)
        generator_adv_loss = adversarial_loss(fake_validity, valid_labels)
        generator_aux_loss = auxiliary_loss(fake_class_pred, generated_labels)
        generator_loss = generator_adv_loss + generator_aux_loss
        
        generator_loss.backward()
        generator_optimizer.step()
        
        # Store losses
        generator_losses.append(generator_loss.item())
        discriminator_losses.append(discriminator_loss.item())
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} "
                  f"D_Loss: {discriminator_loss.item():.4f}, G_Loss: {generator_loss.item():.4f}, "
                  f"Accuracy: {accuracy.item():.2%}")
    
    # Calculate epoch accuracy
    epoch_accuracy = running_accuracy / num_batches
    accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch+1} Average Accuracy: {epoch_accuracy:.2%}")
    
    # Save generated images for each class
    with torch.no_grad():
        fake_images = generator_net(fixed_noise, fixed_labels)
        vutils.save_image(fake_images, f"results/acgan/epoch_{epoch+1:03d}.png", 
                         normalize=True, nrow=10)

# Plot training metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot losses
ax1.plot(generator_losses, label="Generator Loss", alpha=0.7)
ax1.plot(discriminator_losses, label="Discriminator Loss", alpha=0.7)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Loss")
ax1.set_title("ACGAN Training Losses")
ax1.legend()

# Plot accuracy
ax2.plot(accuracies, label="Classification Accuracy", color='green')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Discriminator Classification Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("results/acgan/training_metrics.pdf")
plt.show()

# Display and save real vs fake image comparison
print("Creating real vs fake comparison...")

# Get real images with their labels
real_batch = next(iter(train_loader))
real_images_display = real_batch[0].to(device)[:32]
real_labels_display = real_batch[1].to(device)[:32]

# Generate fake images with same class distribution
with torch.no_grad():
    noise = torch.randn(32, noise_dimension, 1, 1, device=device)
    # Use same labels as real images for fair comparison
    fake_images_display = generator_net(noise, real_labels_display)

# Create comparison figure
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
plt.title("ACGAN Generated Images (Conditioned)")
fake_grid = vutils.make_grid(fake_images_display, padding=5, normalize=True).cpu()
plt.imshow(np.transpose(fake_grid, (1, 2, 0)))

plt.savefig("results/acgan/real_vs_fake_comparison.pdf")
plt.show()

# Save samples
vutils.save_image(real_images_display, "results/acgan/real_samples.png", normalize=True)
vutils.save_image(fake_images_display, "results/acgan/fake_samples.png", normalize=True)

# Generate samples for each class
print("\nGenerating samples for each class...")
samples_per_class = 10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

for class_idx in range(num_classes):
    # Generate multiple samples of the same class
    noise = torch.randn(samples_per_class, noise_dimension, 1, 1, device=device)
    labels = torch.full((samples_per_class,), class_idx, device=device)
    
    with torch.no_grad():
        fake_images = generator_net(noise, labels)
        vutils.save_image(fake_images, f"results/acgan/class_{class_names[class_idx]}.png", 
                         normalize=True, nrow=5)

# Create class-wise comparison
print("Creating class-wise comparison...")
fig, axes = plt.subplots(2, num_classes, figsize=(20, 4))

for i in range(num_classes):
    # Generate one sample per class
    noise = torch.randn(1, noise_dimension, 1, 1, device=device)
    label = torch.tensor([i], device=device)
    
    with torch.no_grad():
        fake_img = generator_net(noise, label)
    
    # Find a real image of this class
    for real_img, real_label in train_loader:
        idx = (real_label == i).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            real_sample = real_img[idx[0]]
            break
    
    # Display real
    axes[0, i].imshow(np.transpose(real_sample * 0.5 + 0.5, (1, 2, 0)))
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Real', rotation=90, size='large')
    
    # Display fake
    fake_display = fake_img.squeeze().cpu() * 0.5 + 0.5
    axes[1, i].imshow(np.transpose(fake_display, (1, 2, 0)))
    axes[1, i].axis('off')
    axes[1, i].set_xlabel(class_names[i])
    if i == 0:
        axes[1, i].set_ylabel('Fake', rotation=90, size='large')

plt.suptitle('Real vs Generated Images by Class')
plt.tight_layout()
plt.savefig("results/acgan/class_comparison.pdf")
plt.show()

# Save final models
torch.save(generator_net.state_dict(), "results/acgan/generator.pth")
torch.save(discriminator_net.state_dict(), "results/acgan/discriminator.pth")

print("ACGAN training completed!")
print("Check results/acgan/ folder for generated images and comparisons")
print("Class-specific images saved as class_[classname].png")
