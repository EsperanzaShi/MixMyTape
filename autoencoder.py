import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        # Increased channels, strided convs for downsampling (only in time)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=(1,2), padding=2)   # (32, n_mels, T1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1)  # (64, n_mels, T2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(1,2), padding=1) # (128, n_mels, T3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=(1,2), padding=1) # (256, n_mels, T4)
        # The time dimension after 4 stride-2 layers on input length 129 is always 9
        self.fc = nn.Linear(256 * n_mels * 9, latent_dim)

    def forward(self, x):
        s1 = F.relu(self.conv1(x))  # (32, n_mels, T1)
        s2 = F.relu(self.conv2(s1)) # (64, n_mels, T2)
        s3 = F.relu(self.conv3(s2)) # (128, n_mels, T3)
        s4 = F.relu(self.conv4(s3)) # (256, n_mels, 9)
        z = s4.view(s4.size(0), -1)  # Flatten all dimensions except batch
        z = self.fc(z)
        return z, [s1, s2, s3, s4]

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, n_mels=80):
        super().__init__()
        self.n_mels = n_mels
        self.fc = nn.Linear(latent_dim, 256 * n_mels * 9)
        self.unflatten = nn.Unflatten(1, (256, n_mels, 9))
        # Transposed convs for upsampling (only in time)
        self.deconv4 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1)) # (128, n_mels, 17)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1))  # (64, n_mels, 33)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1))  # (32, n_mels, 65)
        self.deconv1 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=(1,2), padding=2, output_padding=(0,1))    # (1, n_mels, 129)

    def forward(self, z, skips):
        x = self.fc(z)
        x = self.unflatten(x)
        # U-Net skip connections (concat along channel dim)
        x = torch.cat([x, skips[3]], dim=1)  # (256+256, n_mels, 9)
        x = F.relu(self.deconv4(x))          # (128, n_mels, 17)

        # After deconv4
        if x.shape[-1] > skips[2].shape[-1]:
            x = x[..., :skips[2].shape[-1]]
            skips2 = skips[2]
        elif x.shape[-1] < skips[2].shape[-1]:
            skips2 = skips[2][..., :x.shape[-1]]
        else:
            skips2 = skips[2]
        x = torch.cat([x, skips2], dim=1)
        x = F.relu(self.deconv3(x))  # (64, n_mels, 33)

        # After deconv3
        if x.shape[-1] > skips[1].shape[-1]:
            x = x[..., :skips[1].shape[-1]]
            skips1 = skips[1]
        elif x.shape[-1] < skips[1].shape[-1]:
            skips1 = skips[1][..., :x.shape[-1]]
        else:
            skips1 = skips[1]
        x = torch.cat([x, skips1], dim=1)
        x = F.relu(self.deconv2(x))  # (32, n_mels, 65)

        # After deconv2
        if x.shape[-1] > skips[0].shape[-1]:
            x = x[..., :skips[0].shape[-1]]
            skips0 = skips[0]
        elif x.shape[-1] < skips[0].shape[-1]:
            skips0 = skips[0][..., :x.shape[-1]]
        else:
            skips0 = skips[0]
        x = torch.cat([x, skips0], dim=1)
        x = self.deconv1(x)  # (1, n_mels, 129) -- removed sigmoid
        # Crop or pad to (1, n_mels, 129)
        if x.shape[-1] < 129:
            pad = 129 - x.shape[-1]
            x = F.pad(x, (0, pad))
        elif x.shape[-1] > 129:
            x = x[..., :129]
        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, n_mels=80):
        super().__init__()
        self.encoder = Encoder(latent_dim, n_mels)
        self.decoder = Decoder(latent_dim, n_mels)

    def forward(self, x):
        z, skips = self.encoder(x)
        x_recon = self.decoder(z, skips)
        x_recon = torch.clamp(x_recon, min=-11.5129, max=2.0)  # Clamp output to log-mel range
        return x_recon

    def load_encoder_weights(self, classifier_ckpt_path):
        """Load encoder weights from a pretrained classifier checkpoint (for first 2 conv layers only)."""
        state_dict = torch.load(classifier_ckpt_path, map_location='cpu')
        encoder_dict = self.encoder.state_dict()
        mapping = {
            'conv1.weight': 'conv1.weight',
            'conv1.bias': 'conv1.bias',
            'conv2.weight': 'conv2.weight',
            'conv2.bias': 'conv2.bias',
        }
        for k_src, k_dst in mapping.items():
            if k_src in state_dict:
                encoder_dict[k_dst] = state_dict[k_src]
        self.encoder.load_state_dict(encoder_dict, strict=False) 