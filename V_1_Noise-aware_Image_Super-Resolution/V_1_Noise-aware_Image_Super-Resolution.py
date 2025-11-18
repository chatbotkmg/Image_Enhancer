
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import torchvision.models as models
import torch.nn.functional as F


# =====================================
# Dataset Loader
# =====================================
class SuperResolutionDataset(Dataset):
    
    #Custom dataset for loading paired low-resolution and high-resolution cat image data.
    #Assumes files in both folders match in sorted name order.
    
    def __init__(self, lowres_dir, highres_dir):
        self.lowres_dir = lowres_dir
        self.highres_dir = highres_dir

        # Load only JPG files and sort to ensure correct pairing
        self.low_files = sorted([f for f in os.listdir(lowres_dir) if f.endswith('.jpg')])
        self.high_files = sorted([f for f in os.listdir(highres_dir) if f.endswith('.jpg')])

        # Convert images to tensors, keep value range 0–1 (no normalization)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.low_files)

    def __getitem__(self, idx):
        # Fetch corresponding image paths
        low_path = os.path.join(self.lowres_dir, self.low_files[idx])
        high_path = os.path.join(self.highres_dir, self.high_files[idx])

        # Load as RGB
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # No resizing applied here — images are already saved as 128×128 and 512×512 before loading
        low_tensor = self.to_tensor(low_img)
        high_tensor = self.to_tensor(high_img)

        return low_tensor, high_tensor


# Dataset configuration
lowres_dir = "/kaggle/working/generatored"
highres_dir = "/kaggle/working/highres"

dataset = SuperResolutionDataset(lowres_dir, highres_dir)

dataloader = DataLoader(
    dataset,
    batch_size=8,                  # Larger batch for stable gradient estimation
    shuffle=True,                 # Randomize sample order
    num_workers=2,                # Use 2 workers to speed up IO
    pin_memory=True
)


# =====================================
# Model Building Blocks
# =====================================

class ConvBlock(nn.Module):
    
    #Two convolution layers with LeakyReLU.
    #Used across the network for basic feature extraction.
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    
    #Downsampling: MaxPool followed by convolution block.
    #Simulates UNet-like encoder contraction path.
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    
    #Upsampling via ConvTranspose2d + skip connection.
    #Skip connection helps preserve spatial detail (early resolution features).
    
    def __init__(self, up_in_channels, skip_in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)  # Merge encoder features
        return self.conv(x)


class DetailEnhanceBlock(nn.Module):
    
    #Residual refinement block to boost image detail via multiple layers of convolution.
    #Helps recover texture that simple UNet + PixelShuffle may miss.
    
    def __init__(self, channels, num_layers=5):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)  # Residual connection


# =====================================
# Super-Resolution Model (UNet + PixelShuffle + Enhance)
# =====================================
class UNetSuperResEnhanced(nn.Module):
    
    #UNet-based encoder-decoder with PixelShuffle for upscaling and additional
    #detail enhancement block. Final upscale factor is ×4.
    
    def __init__(self):
        super().__init__()

        # Encoder
        self.c1 = ConvBlock(3, 64)
        self.d1 = DownBlock(64, 128)
        self.d2 = DownBlock(128, 256)
        self.d3 = DownBlock(256, 512)
        self.d4 = DownBlock(512, 512)
        self.d5 = DownBlock(512, 512)

        # Decoder
        self.u4 = UpBlock(512, 512, 512)
        self.u3 = UpBlock(512, 512, 256)
        self.u2 = UpBlock(256, 256, 128)
        self.u1 = UpBlock(128, 128, 64)
        self.u0 = UpBlock(64, 64, 64)

        # PixelShuffle prepares final upscale
        self.final_conv = nn.Conv2d(64, 3 * 16, kernel_size=1)  # (4×4 upscale)
        self.pixel_shuffle = nn.PixelShuffle(4)

        # Optional feature refinement
        self.detail_enhance = DetailEnhanceBlock(3, num_layers=5)

    def forward(self, x):
        # Encoder path
        s1 = self.c1(x)
        s2 = self.d1(s1)
        s3 = self.d2(s2)
        s4 = self.d3(s3)
        s5 = self.d4(s4)
        b  = self.d5(s5)

        # Decoder path
        x = self.u4(b, s5)
        x = self.u3(x, s4)
        x = self.u2(x, s3)
        x = self.u1(x, s2)
        x = self.u0(x, s1)

        # PixelShuffle and enhancement
        x = self.final_conv(x)
        x = self.pixel_shuffle(x)
        x = self.detail_enhance(x)

        return x


# =====================================
# Weight Initialization
# =====================================
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# =====================================
# Perceptual and Edge Loss
# =====================================

class VGGFeatureExtractor(nn.Module):
    
    #Extract intermediate VGG16 layers for high-level feature comparison.
    #Used in perceptual loss.
    
    def __init__(self, device='cuda', layers=(2, 7, 14, 21)):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layers = layers

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    
    #Loss based on VGG feature similarity.
    #Encourages high-level texture resemblance beyond pixel accuracy.
   
    def __init__(self, device='cuda'):
        super().__init__()
        self.extractor = VGGFeatureExtractor(device)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        fp = self.extractor(pred)
        ft = self.extractor(target)
        loss = sum(self.l1(a, b) for a, b in zip(fp, ft))
        return loss


class EdgeLoss(nn.Module):
    
    #Edge-based loss using Sobel filters.
    #Helps preserve sharp structures such as cat facial contours.
    
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.l1 = nn.L1Loss()

    def gradient(self, img):
        grads = []
        for c in range(img.shape[1]):
            ch = img[:, c:c+1]
            gx = F.conv2d(ch, self.kx, padding=1)
            gy = F.conv2d(ch, self.ky, padding=1)
            g = torch.sqrt(gx**2 + gy**2 + 1e-6)
            grads.append(g)
        return torch.cat(grads, dim=1)

    def forward(self, pred, target):
        gp = self.gradient(pred)
        gt = self.gradient(target)
        return self.l1(gp, gt)


# =====================================
# Combined Loss (Two-stage training strategy)
# =====================================
class CombinedLoss(nn.Module):
    
    #Final loss = L1 + 0.02 * Perceptual + 0.5 * Edge
    #Used only after initial training with pure L1.
    
    def __init__(self, device='cuda', w_l1=1.0, w_perc=0.02, w_edge=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss(device)
        self.edge = EdgeLoss()
        self.w_l1 = w_l1
        self.w_perc = w_perc
        self.w_edge = w_edge

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_p = self.perc(pred, target)
        loss_e = self.edge(pred, target)

        total = (self.w_l1 * loss_l1 +
                 self.w_perc * loss_p +
                 self.w_edge * loss_e)

        return total, {"l1": loss_l1.item(), "perc": loss_p.item(), "edge": loss_e.item()}


# =====================================
# Visualization
# =====================================
def visualize_batch(low_res, high_res, output, epoch, save_path=None):
    
    #Display & optionally save low-res input, enhanced output, and high-res target.
    #Useful for periodic inspection during training.
    
    lr_img = low_res[0].cpu().permute(1,2,0).numpy()
    hr_img = high_res[0].cpu().permute(1,2,0).numpy()
    out_img = output[0].cpu().permute(1,2,0).detach().numpy()

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(lr_img); axes[0].set_title('Low-res'); axes[0].axis('off')
    axes[1].imshow(out_img); axes[1].set_title('Output'); axes[1].axis('off')
    axes[2].imshow(hr_img); axes[2].set_title('High-res'); axes[2].axis('off')

    plt.suptitle(f'Epoch {epoch}')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


# =====================================
# Training Loop
# =====================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetSuperResEnhanced().to(device)
initialize_weights(model)

criterion_l1 = nn.L1Loss()  # Phase 1: only L1 loss
criterion_combined = CombinedLoss(device=device).to(device)  # Phase 2: final loss

optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 300

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0

    for i, (lr, hr) in enumerate(dataloader):
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        output = model(lr)

        # =============================
        # Loss Strategy:
        #   Epoch 1–100  → Pure L1
        #   Epoch 101+   → Full Combined Loss
        # =============================
        if epoch <= 100:
            loss = criterion_l1(output, hr)
            loss_parts = {"L1": loss.item()}   # For logging only
        else:
            loss, loss_parts = criterion_combined(output, hr)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

    # Save visualization every 5 epochs
    if epoch % 5 == 0:
        lr, hr = next(iter(dataloader))
        lr, hr = lr.to(device), hr.to(device)
        with torch.no_grad():
            output = model(lr)

        save_dir1 = '/kaggle/working/V_1_Noise-aware_Image_Super-Resolution/generatored'
        os.makedirs(save_dir1, exist_ok=True)
        save_path = f"{save_dir1}/epoch_{epoch}.png"
        visualize_batch(lr, hr, output, epoch, save_path=save_path)




