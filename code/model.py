import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed # For embedding patches in ViT style
# Ensure torch_geometric is installed for GAT
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data # For creating graph batches
except ImportError:
    raise ImportError("torch_geometric not found. Please install it with pip install torch_geometric")


class CustomConv1(nn.Module):
    """
    Handles varying input channels for pre-trained models.
    Initializes new channels by averaging/replicating existing 3-channel weights.
    """
    def __init__(self, original_conv1: nn.Conv2d, num_input_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            num_input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # Initialize new conv1 weights
        if num_input_channels != original_conv1.in_channels:
            # Replicate/average pre-trained weights for new input channels
            # Example: average across RGB channels for new channels
            if original_conv1.in_channels == 3:
                # Take mean of the 3 channels and repeat for extra channels
                weight = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, num_input_channels, 1, 1)
                # Or for more direct replication of first 3 and random for others:
                # weight = torch.randn(self.conv.weight.shape) * 0.01 # Start with small random
                # weight[:, :original_conv1.in_channels, :, :] = original_conv1.weight
                self.conv.weight = nn.Parameter(weight)
                if original_conv1.bias is not None:
                    self.conv.bias = nn.Parameter(original_conv1.bias)
            else: # If original was not 3, just initialize normally
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                if self.conv.bias is not None:
                    nn.init.zeros_(self.conv.bias)
        else: # If input channels match, just copy original weights
            self.conv.weight = original_conv1.weight
            if original_conv1.bias is not None:
                self.conv.bias = original_conv1.bias

    def forward(self, x):
        return self.conv(x)


class CNNEncoder(nn.Module):
    def __init__(self, model_name='resnet50', num_input_channels=3, pretrained=True):
        super().__init__()
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")

        # Modify conv1 for input channels
        if num_input_channels != 3:
            backbone.conv1 = CustomConv1(backbone.conv1, num_input_channels)
            # If pretrained weights are used with non-3 channels, need to handle loading
            # For simplicity, we assume CustomConv1 handles initial weight transfer.
            # If 'pretrained=True' was used above, CustomConv1 would have received the original conv1 weights
            # and adapted them. Other layers remain loaded.

        # Extract stages for skip connections
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1 # ~1/4 spatial resolution
        self.layer2 = backbone.layer2 # ~1/8 spatial resolution
        self.layer3 = backbone.layer3 # ~1/16 spatial resolution
        self.layer4 = backbone.layer4 # ~1/32 spatial resolution

        # Determine feature dimensions for later use
        self.out_channels = [
            backbone.layer1[-1].conv3.out_channels, # For layer1 output
            backbone.layer2[-1].conv3.out_channels, # For layer2 output
            backbone.layer3[-1].conv3.out_channels, # For layer3 output
            backbone.layer4[-1].conv3.out_channels  # For layer4 output
        ]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1) # Output for skip connection
        x3 = self.layer2(x2) # Output for skip connection
        x4 = self.layer3(x3) # Output for skip connection
        x5 = self.layer4(x4) # Highest level features for ViT/GAT
        return x2, x3, x4, x5 # Return multi-scale features


class HybridViTGATBlock(nn.Module):
    """
    A block that combines ViT-like self-attention (or cross-attention)
    and Graph Attention Network for integrated spatial-global reasoning.
    Inspired by GFormer's Hybrid-ViT blocks.
    """
    def __init__(self, in_channels, embed_dim, num_heads, gat_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        
        # Linear projection from CNN features to ViT embed_dim
        self.proj = nn.Linear(in_channels, embed_dim)

        # ViT-like self-attention (or cross-attention if multiple inputs)
        # For simplicity, let's use standard self-attention within this block initially.
        # A more complex cross-attention would take another tensor as input.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gat_proc_linear = nn.Linear(embed_dim, embed_dim) # Project to GAT input dim if needed
        self.gat_conv = GATConv(embed_dim, embed_dim // gat_heads, heads=gat_heads, dropout=dropout, concat=True) # Concatenate heads
        
        self.norm3 = nn.LayerNorm(embed_dim * gat_heads) # Norm after GAT concat
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * gat_heads, embed_dim), # Reduce back to embed_dim
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(embed_dim)


    def forward(self, x_cnn_feat):
        # x_cnn_feat: (B, C, H, W)
        B, C, H, W = x_cnn_feat.shape
        
        # 1. Prepare tokens for ViT part
        # Flatten spatial dimensions to sequence of tokens
        # (B, C, H, W) -> (B, H*W, C)
        x_tokens = x_cnn_feat.flatten(2).permute(0, 2, 1) # (B, N_patches, C_in)
        
        # Normalize before projection
        x_tokens = self.norm1(x_tokens)
        
        # Project to embed_dim
        x_vit_input = self.proj(x_tokens) # (B, N_patches, embed_dim)
        
        # Add positional encoding (can be learned or fixed)
        # For this example, assuming external positional encoding or adding it in a wrapper.
        # Here, it's implicitly handled as part of the attention if `pos_embed` is added externally.
        
        # 2. ViT-like Self-Attention
        attn_output, _ = self.attn(x_vit_input, x_vit_input, x_vit_input)
        x_after_attn = x_vit_input + attn_output # Residual connection
        
        # 3. GAT Layer (on the same tokens, but modeling relations)
        # Reshape to (N_patches*B, embed_dim) for PyG GATConv input
        num_nodes_per_batch = H * W
        node_features_flat = x_after_attn.reshape(-1, x_after_attn.shape[-1]) # (B*N, E)
        
        # Create grid-like edge index for GAT (for simplicity; can be dynamic)
        # For a full batch, need to offset node indices per graph in the batch
        edge_index_list = []
        batch_indices = []
        for i in range(B):
            node_offset = i * num_nodes_per_batch
            # Add horizontal edges
            for r in range(H):
                for c in range(W - 1):
                    idx1 = r * W + c + node_offset
                    idx2 = r * W + (c + 1) + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            # Add vertical edges
            for r in range(H - 1):
                for c in range(W):
                    idx1 = r * W + c + node_offset
                    idx2 = (r + 1) * W + c + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            # Add diagonal edges (optional, but enhances connectivity)
            for r in range(H - 1):
                for c in range(W - 1):
                    idx1 = r * W + c + node_offset
                    idx2_diag1 = (r + 1) * W + (c + 1) + node_offset
                    idx2_diag2 = (r + 1) * W + (c - 1) + node_offset if c > 0 else -1
                    if idx2_diag1 != -1: edge_index_list.extend([[idx1, idx2_diag1], [idx2_diag1, idx1]])
                    if idx2_diag2 != -1: edge_index_list.extend([[idx1, idx2_diag2], [idx2_diag2, idx1]])

            batch_indices.extend([i] * num_nodes_per_batch) # PyG Data.batch equivalent
            
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(node_features_flat.device)
        
        x_gat_input = self.norm2(node_features_flat) # Normalize before GAT
        x_gat_input = self.gat_proc_linear(x_gat_input) # Project to GAT input dim
        
        x_gat = self.gat_conv(x_gat_input, edge_index) # Output (N*B, embed_dim*gat_heads)
        x_gat = F.elu(x_gat) # Activation after GAT
        
        # 4. MLP (Feed-Forward Network)
        x_after_gat = x_after_attn.reshape(-1, x_after_attn.shape[-1]) + x_gat # Residual + GAT output (align dims)
        x_after_gat_norm = self.norm3(x_after_gat)
        
        mlp_output = self.mlp(x_after_gat_norm)
        x_final_block_output_flat = x_after_gat_norm + mlp_output # Residual connection
        x_final_block_output_flat = self.final_norm(x_final_block_output_flat)

        # Reshape back to spatial (B, embed_dim, H, W)
        x_final_block_output = x_final_block_output_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_final_block_output


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        x = self.upsample(x)
        # Ensure sizes match before concatenation (important for different resolutions)
        if x.size() != skip_features.size():
            x = F.interpolate(x, size=skip_features.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_relu(x)
        return x


class HybridGeoNet(nn.Module):
    def __init__(self, num_input_channels=8, num_classes=1,
                 cnn_model='resnet50', pretrained_cnn=True,
                 vit_embed_dim=768, vit_depth=4, vit_heads=12, gat_heads=4):
        super().__init__()
        self.num_classes = num_classes

        # 1. CNN Encoder
        self.cnn_encoder = CNNEncoder(cnn_model, num_input_channels, pretrained_cnn)

        # Get output channels for relevant encoder stages
        # ResNet stages: layer1 (C=256), layer2 (C=512), layer3 (C=1024), layer4 (C=2048)
        # Let's apply ViT-GAT on the highest resolution features (layer4 output)
        cnn_high_level_channels = self.cnn_encoder.out_channels[3] # 2048 for ResNet50/101

        # 2. Hybrid-ViT-GAT Block
        # Applying a single block on the highest-level CNN features for simplicity,
        # but can be multiple stacked blocks.
        self.hybrid_vit_gat_block = HybridViTGATBlock(
            in_channels=cnn_high_level_channels,
            embed_dim=vit_embed_dim,
            num_heads=vit_heads,
            gat_heads=gat_heads
        )
        # A separate linear layer to project the ViT-GAT output to a usable channel count for fusion
        self.vit_gat_proj = nn.Conv2d(vit_embed_dim, 512, kernel_size=1) # Project to 512 channels for fusion

        # 3. CNN Decoder with Skip Connections and Global Feature Fusion
        # Start decoding from the output of the Hybrid-ViT-GAT block
        # Fusing the projected ViT-GAT output with cnn_encoder.layer4 output
        
        # Determine initial decoder input channels after fusion (cnn_high_level_channels + vit_gat_proj_channels)
        # Let's refine the fusion: take the output of layer4, pass it through ViT-GAT.
        # The ViT-GAT output will be the "enhanced high-level features".
        # Then, proceed with a UNet-like decoder.
        
        # Initial Conv for decoder, takes the enhanced features from ViT-GAT
        self.decoder_entry_conv = nn.Conv2d(vit_embed_dim, 1024, kernel_size=3, padding=1) # Assuming vit_embed_dim is input from block

        # Decoder blocks: upsample and concatenate with skip connections
        # Corresponding skip channels from CNNEncoder: layer3 (1024), layer2 (512), layer1 (256)
        self.upconv4 = DecoderBlock(1024, self.cnn_encoder.out_channels[2], 512) # from decoder_entry_conv to layer3 skip
        self.upconv3 = DecoderBlock(512, self.cnn_encoder.out_channels[1], 256)  # from upconv4 to layer2 skip
        self.upconv2 = DecoderBlock(256, self.cnn_encoder.out_channels[0], 128) # from upconv3 to layer1 skip
        
        # Last upsampling to original input resolution scale (e.g. 224x224 after conv1+maxpool for 1/4 res)
        # Assuming original input H, W are multiples of 32 (ResNet layer4 output is 1/32)
        # The DecoderBlock upsamples by factor of 2. 3 blocks = 2^3 = 8x upsampling.
        # If layer1 is 1/4, and layer4 is 1/32, then we need 3 upsampling steps to reach 1/4.
        # And another upsampling step to reach full resolution after layer1.
        
        # Adjusting decoder stages based on typical ResNet/UNet connections:
        # Layer4 (1/32) -> ViT-GAT -> (Output is at 1/32 scale)
        # Decoder 1: 1/32 -> 1/16 (uses skip from layer3)
        # Decoder 2: 1/16 -> 1/8 (uses skip from layer2)
        # Decoder 3: 1/8 -> 1/4 (uses skip from layer1)
        # Decoder 4: 1/4 -> 1/1 (uses initial conv1 block features if needed, or simple final upsampling)
        
        # Let's adjust decoder structure to be more explicit:
        self.decoder_layer4 = nn.Sequential(
            nn.Conv2d(vit_embed_dim, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_layer3 = DecoderBlock(1024, self.cnn_encoder.out_channels[2], 512) # Skip from layer3
        self.decoder_layer2 = DecoderBlock(512, self.cnn_encoder.out_channels[1], 256) # Skip from layer2
        self.decoder_layer1 = DecoderBlock(256, self.cnn_encoder.out_channels[0], 128) # Skip from layer1

        # Final upsample to original resolution
        # From 1/4 resolution (output of decoder_layer1) to original resolution
        self.final_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv_relu = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        B, C_in, H_in, W_in = x.shape

        # CNN Encoder
        x2, x3, x4, x5 = self.cnn_encoder(x) # x2(1/4), x3(1/8), x4(1/16), x5(1/32)

        # Hybrid-ViT-GAT Block
        # Process the deepest CNN features (x5 is Bx2048xH/32xW/32)
        vit_gat_features = self.hybrid_vit_gat_block(x5) # Bxvit_embed_dimxH/32xW/32

        # Decoder path with skip connections
        # Start decoding from the enhanced features
        x = self.decoder_layer4(vit_gat_features) # Bx1024xH/32xW/32

        x = self.decoder_layer3(x, x4) # Input: 1024, Skip: x4(1024), Output: 512. Size: H/16xW/16
        x = self.decoder_layer2(x, x3) # Input: 512, Skip: x3(512), Output: 256. Size: H/8xW/8
        x = self.decoder_layer1(x, x2) # Input: 256, Skip: x2(256), Output: 128. Size: H/4xW/4

        # Final upsampling to original resolution
        x = self.final_upsample(x) # Size: H/2xW/2 (or HxW if input was 1/2 res after initial maxpool)
        
        # If initial CNN output was 1/4, this brings it to 1/2. Need one more upsample.
        # Assuming typical ResNet, output of conv1+maxpool (x1) is 1/4 of original.
        # So x2 is 1/4, x3 1/8, x4 1/16, x5 1/32.
        # Decoder 1/32 -> 1/16 (x4)
        # Decoder 1/16 -> 1/8 (x3)
        # Decoder 1/8 -> 1/4 (x2)
        # We need to upsample from H/4xW/4 to H_inxW_in. This requires x1 or initial input features.
        
        # Let's ensure the final upsampling is correct.
        # Assuming original input image is H_in x W_in
        # x2 is H_in/4 x W_in/4
        # after decoder_layer1, x is H_in/4 x W_in/4, channels 128
        
        # One more upsampling step to reach original resolution
        # If the input was 224x224, conv1+maxpool makes it 56x56.
        # x2 is 56x56. So decoder_layer1 output is 56x56.
        # To get 224x224, we need two more 2x upsamples (56->112->224).
        
        x = F.interpolate(x, size=(H_in, W_in), mode='bilinear', align_corners=False)
        x = self.final_conv_relu(x)
        
        output_mask = self.output_conv(x)
        return output_mask