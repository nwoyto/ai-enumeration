import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer # Although imported, VisionTransformer itself isn't directly used in this specific architecture
from timm.models.layers import PatchEmbed # Although imported, PatchEmbed isn't directly used in this specific architecture

# Ensure torch_geometric is installed for GAT functionalities.
# If not installed, a dummy GATConv and Data class are provided to allow the code to run,
# but actual GAT operations will be replaced by a simple linear layer.
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data # For creating graph batches within the HybridViTGATBlock
    PYG_AVAILABLE = True
except ImportError:
    print("WARNING: torch_geometric not found. Using dummy GATConv and Data classes.")
    PYG_AVAILABLE = False
    class GATConv(nn.Module):
        """
        A dummy GATConv implementation for when torch_geometric is not available.
        It simply acts as a linear layer.
        """
        def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, add_self_loops=True, bias=True):
            super().__init__()
            # The linear layer projects input features to the expected output dimension.
            # If concat=True, output dimension is out_channels * heads; otherwise, it's out_channels.
            self.linear = nn.Linear(in_channels, out_channels * heads if concat else out_channels)
            print("NOTE: Dummy GATConv is a linear layer. Install 'torch_geometric' for actual GAT functionality.")
        def forward(self, x, edge_index):
            # In a real GAT, edge_index would define graph connectivity. Here, it's ignored.
            return self.linear(x)
    class Data(object):
        """
        A dummy Data class for compatibility when torch_geometric.data.Data is not available.
        """
        def __init__(self, x, edge_index, batch=None):
            self.x = x
            self.edge_index = edge_index
            self.batch = batch


# --- Custom Modules ---

class CustomConv1(nn.Module):
    """
    Handles varying input channels for pre-trained CNN models' initial convolution layer (conv1).
    Pre-trained models (like ResNet) are typically trained on 3-channel RGB images.
    If the input `num_input_channels` is different, this module adjusts the `conv1` layer.
    For `num_input_channels > 3`, it initializes new channels by averaging the weights of the
    original 3 input channels and replicating them, or by Kaiming initialization if `original_conv1`
    wasn't 3 channels to begin with.
    """
    def __init__(self, original_conv1: nn.Conv2d, num_input_channels: int):
        super().__init__()
        # Create a new Conv2d layer with the desired number of input channels.
        self.conv = nn.Conv2d(
            num_input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=True if original_conv1.bias is not None else False # Ensure bias is handled correctly
        )
        
        # If the number of input channels for the new conv layer is different from the original,
        # we need to adjust its weights.
        if num_input_channels != original_conv1.in_channels:
            # Case 1: Original conv1 was 3 channels (typical for pre-trained ImageNet models).
            # We average the 3 input channel weights and then repeat them for the new 'num_input_channels'.
            if original_conv1.in_channels == 3:
                # `mean(dim=1, keepdim=True)` averages across the input channels, resulting in (out_channels, 1, kH, kW).
                # `repeat(1, num_input_channels, 1, 1)` replicates this single-channel weight
                # across the `num_input_channels` dimensions.
                weight = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, num_input_channels, 1, 1)
                self.conv.weight = nn.Parameter(weight)
                # Copy bias if it exists
                if original_conv1.bias is not None:
                    self.conv.bias = nn.Parameter(original_conv1.bias)
            # Case 2: Original conv1 was not 3 channels (less common for pre-trained backbones).
            # Initialize with Kaiming Normal (He initialization) for ReLU activation,
            # which is suitable for standard convolutional layers.
            else: 
                nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
                if self.conv.bias is not None:
                    nn.init.zeros_(self.conv.bias)
        # If the number of input channels matches the original, simply use the original weights and bias.
        else:
            self.conv.weight = original_conv1.weight
            if original_conv1.bias is not None:
                self.conv.bias = original_conv1.bias

    def forward(self, x):
        return self.conv(x)


class CNNEncoder(nn.Module):
    """
    A Convolutional Neural Network (CNN) backbone used as an encoder for feature extraction.
    It's designed to provide multi-scale features, similar to the encoder part of a U-Net,
    which are then used for skip connections in the decoder.
    Supports ResNet50 and ResNet101 as backbones.
    """
    def __init__(self, model_name='resnet50', num_input_channels=3, pretrained=True):
        super().__init__()
        # Load the specified pre-trained ResNet model.
        if model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")

        # Adjust the first convolutional layer (conv1) if the input channel count differs from 3.
        if num_input_channels != 3:
            original_conv1_layer = backbone.conv1
            backbone.conv1 = CustomConv1(original_conv1_layer, num_input_channels)

        # Define the sequential layers of the CNN encoder, extracting features at different stages.
        # x1: Output after conv1, bn1, relu, maxpool (typically 1/4th resolution of input)
        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        # x2: Output of ResNet's layer1 (typically 1/4th resolution)
        self.layer1 = backbone.layer1
        # x3: Output of ResNet's layer2 (typically 1/8th resolution)
        self.layer2 = backbone.layer2
        # x4: Output of ResNet's layer3 (typically 1/16th resolution)
        self.layer3 = backbone.layer3
        # x5: Output of ResNet's layer4 (typically 1/32nd resolution - highest-level features)
        self.layer4 = backbone.layer4

        # Store the output channel dimensions for each feature map (x2, x3, x4, x5).
        # These channels will correspond to the skip connections in the U-Net decoder.
        self.out_channels = [
            backbone.layer1[-1].conv3.out_channels, # For x2 (256 for ResNet50/101)
            backbone.layer2[-1].conv3.out_channels, # For x3 (512 for ResNet50/101)
            backbone.layer3[-1].conv3.out_channels, # For x4 (1024 for ResNet50/101)
            backbone.layer4[-1].conv3.out_channels  # For x5 (2048 for ResNet50/101)
        ]

    def forward(self, x):
        # Pass the input through the CNN layers and collect features at each stage.
        x1 = self.conv1(x) # Initial downsampling
        x2 = self.layer1(x1) # Features from block 1
        x3 = self.layer2(x2) # Features from block 2
        x4 = self.layer3(x3) # Features from block 3
        x5 = self.layer4(x4) # Features from block 4 (highest level, lowest resolution)
        return x2, x3, x4, x5 # Return features for skip connections and the highest-level features for the HybridViTGATBlock


class HybridViTGATBlock(nn.Module):
    """
    A block that combines Vision Transformer (ViT)-like self-attention and
    Graph Attention Network (GAT) for enhanced global and local reasoning.
    It processes the highest-level CNN features (x5).
    """
    def __init__(self, in_channels, embed_dim, num_heads, gat_heads, dropout=0.2):
        super().__init__()
        # Layer normalization for the input features from CNN.
        # It expects `in_channels` features (e.g., 2048 from ResNet's layer4).
        self.norm1 = nn.LayerNorm(in_channels)
        # Linear projection to transform CNN features to `embed_dim` for ViT-like attention.
        self.proj = nn.Linear(in_channels, embed_dim) # e.g., 2048 -> 768

        # Multi-head Self-Attention, similar to ViT's attention mechanism.
        # `embed_dim` is the dimension of the input and output features for attention.
        # Double the number of attention heads for more capacity
        self.attn = nn.MultiheadAttention(embed_dim, num_heads * 2, dropout=dropout, batch_first=True)
        
        # Layer normalization before processing for GAT.
        self.norm2 = nn.LayerNorm(embed_dim)
        # Linear layer to prepare features for the GAT. This can be an identity if dimensions match.
        self.gat_proc_linear = nn.Linear(embed_dim, embed_dim) # From embed_dim (768) to embed_dim (768)

        # First GATConv layer.
        # Input: `embed_dim` (768)
        # Output per head: `embed_dim // gat_heads` (768 // 4 = 192)
        # Since `concat=True`, the total output dimension is `(embed_dim // gat_heads) * gat_heads = 192 * 4 = 768`.
        # Double the number of GAT heads for more capacity
        self.gat_conv1 = GATConv(embed_dim, embed_dim // (gat_heads * 2), heads=gat_heads * 2, dropout=dropout, concat=True)
        
        # --- CRITICAL FIX APPLIED HERE ---
        # This linear layer projects the output of the first GATConv layer back to `embed_dim`.
        # The input dimension to this layer should match the actual output dimension of `gat_conv1`,
        # which is `embed_dim` (768) because `out_channels * heads` simplifies to `embed_dim` when `concat=True`.
        # The original code had `embed_dim * gat_heads` as input, which was `3072`, causing the shape mismatch.
        self.proj_gat1_to_embed_dim = nn.Linear(embed_dim, embed_dim) # Corrected: 768 -> 768
        
        # Second GATConv layer.
        # Input: `embed_dim` (768), coming from `proj_gat1_to_embed_dim`.
        # Output per head: `embed_dim // gat_heads` (192)
        # Since `concat=False`, the output dimension is just `out_channels`, which is `embed_dim // gat_heads = 192`.
        self.gat_conv2 = GATConv(embed_dim, embed_dim // (gat_heads * 2), heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

        # Linear layer to project `x_after_attn_residual` (which is `embed_dim`)
        # to match the final GAT output dimension (now `embed_dim // (gat_heads * 2)`) for residual connection.
        self.residual_proj = nn.Linear(embed_dim, embed_dim // (gat_heads * 2))

        # Normalization after the GAT output and residual connection.
        self.norm3 = nn.LayerNorm(embed_dim // (gat_heads * 2))

        # Deepened MLP (Feed-Forward Network) after GAT and residual.
        # It operates on features of dimension `embed_dim // (gat_heads * 2)`.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim // (gat_heads * 2), embed_dim // (gat_heads * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // (gat_heads * 2), embed_dim // (gat_heads * 2) * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // (gat_heads * 2) * 2, embed_dim // (gat_heads * 2)),
            nn.Dropout(dropout)
        )
        # Final layer normalization.
        self.final_norm = nn.LayerNorm(embed_dim // (gat_heads * 2))

        # Projection to match decoder input channels (192 expected)
        self.decoder_proj = nn.Linear(embed_dim // (gat_heads * 2), 192)


    def forward(self, x_cnn_feat):
        B, C, H, W = x_cnn_feat.shape # Batch size, Channels, Height, Width
        num_nodes_per_batch = H * W # Each pixel/feature location becomes a node in the graph.

        # 1. Prepare tokens for ViT part
        # Flatten spatial dimensions (H*W) into sequence length and permute to (B, N_patches, C).
        x_tokens = x_cnn_feat.flatten(2).permute(0, 2, 1) # (B, H*W, C) e.g., (B, 64, 2048) if H=8, W=8
        x_tokens_normed = self.norm1(x_tokens) # Normalize features
        
        # Project features to the embedding dimension for ViT attention.
        x_vit_input = self.proj(x_tokens_normed) # (B, N_patches, embed_dim) e.g., (B, 64, 768)
        
        # 2. ViT-like Self-Attention
        # Perform multi-head self-attention. Queries, Keys, Values are all from `x_vit_input`.
        attn_output, _ = self.attn(x_vit_input, x_vit_input, x_vit_input)
        # Add residual connection: input + attention_output.
        x_after_attn_residual = x_vit_input + attn_output # (B, N_patches, embed_dim=768)
        
        # Reshape to flatten all tokens across the batch into a single sequence for GAT processing.
        # This treats all (B * N_patches) as individual nodes in a large graph.
        node_features_flat = x_after_attn_residual.reshape(-1, x_after_attn_residual.shape[-1]) # (B*N_patches, embed_dim=768)
        
        # 3. GAT Layer
        # Normalize node features before GAT.
        x_gat_input = self.norm2(node_features_flat)
        # Apply a linear transformation for GAT input (can be identity if `embed_dim` matches).
        x_gat_input = self.gat_proc_linear(x_gat_input) # Still (B*N_patches, embed_dim=768)

        # Create grid-like edge index for GAT.
        # This defines connections between adjacent and diagonally adjacent pixels/nodes
        # within each image in the batch.
        edge_index_list = []
        for i in range(B): # Iterate through each image in the batch
            node_offset = i * num_nodes_per_batch # Offset for nodes of the current image
            for r in range(H):
                for c in range(W - 1): # Horizontal connections
                    idx1 = r * W + c + node_offset
                    idx2 = r * W + (c + 1) + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]]) # Add bi-directional edges
            for r in range(H - 1):
                for c in range(W): # Vertical connections
                    idx1 = r * W + c + node_offset
                    idx2 = (r + 1) * W + c + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            for r in range(H - 1):
                for c in range(W - 1): # Diagonal (top-left to bottom-right)
                    idx1 = r * W + c + node_offset
                    idx2_diag1 = (r + 1) * W + (c + 1) + node_offset
                    if idx2_diag1 != -1: edge_index_list.extend([[idx1, idx2_diag1], [idx2_diag1, idx1]])
                for c in range(1, W): # Diagonal (top-right to bottom-left, c > 0)
                    idx1 = r * W + c + node_offset
                    idx2_diag2 = (r + 1) * W + (c - 1) + node_offset
                    if idx2_diag2 != -1: edge_index_list.extend([[idx1, idx2_diag2], [idx2_diag2, idx1]])
            
        # Convert the list of edges to a PyTorch tensor in the required format (2, num_edges).
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(node_features_flat.device)
        
        # First GATConv layer. Output is (B*N_patches, embed_dim) (768).
        x_gat_1_output = self.gat_conv1(x_gat_input, edge_index)
        x_gat_1_output = F.elu(x_gat_1_output) # ELU activation
        x_gat_1_output = self.dropout(x_gat_1_output)

        # Project output of first GAT back to `embed_dim` (768) before the second GAT.
        # This is where the fix was applied to match the input dimension.
        x_gat_2_input = self.proj_gat1_to_embed_dim(x_gat_1_output) # (B*N_patches, embed_dim=768)
        
        # Second GATConv layer. Output is (B*N_patches, embed_dim // gat_heads) (192).
        x_gat_final_output = self.gat_conv2(x_gat_2_input, edge_index)
        
        # 4. MLP (Feed-Forward Network) after GAT and Residual Connection
        # Project the original `x_after_attn_residual` (from ViT part) to match the dimension
        # of the GAT output for a residual connection.
        x_after_attn_projected_for_residual = self.residual_proj(x_after_attn_residual.reshape(-1, x_after_attn_residual.shape[-1])) # (B*N_patches, 192)
        
        # Add the GAT output to the projected ViT features for a residual connection.
        x_after_gat_and_residual = x_after_attn_projected_for_residual + x_gat_final_output # Both are (B*N_patches, 192)
        
        # Normalize the combined features.
        x_after_gat_norm = self.norm3(x_after_gat_and_residual) # (B*N_patches, 192)
        
        # Pass through the MLP (Feed-Forward Network).
        mlp_output = self.mlp(x_after_gat_norm) # (B*N_patches, 192)
        
        # Add residual connection for the MLP.
        x_final_block_output_flat = x_after_gat_norm + mlp_output # (B*N_patches, 192)
        x_final_block_output_flat = self.final_norm(x_final_block_output_flat) # (B*N_patches, 192)

        # Reshape the flat output back to a 4D tensor (Batch, Channels, Height, Width).
        x_final_block_output = x_final_block_output_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2) # (B, 192, H, W)
        return x_final_block_output


class DecoderBlock(nn.Module):
    """
    Standard U-Net style decoder block. It performs upsampling, concatenates with
    corresponding skip features from the encoder, and then applies two convolutional layers.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Transposed convolution for upsampling.
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Sequential block of two Conv2d layers with ReLU activation.
        # The input channels to the first Conv2d are `out_channels` (from upsample) + `skip_channels`.
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        # Upsample the input feature map.
        x = self.upsample(x)
        # Ensure that the upsampled feature map has the same spatial dimensions
        # as the skip features for concatenation. This handles potential minor dimension mismatches.
        if x.size() != skip_features.size():
            x = F.interpolate(x, size=skip_features.size()[2:], mode='bilinear', align_corners=False)
        # Concatenate upsampled features with skip features along the channel dimension.
        x = torch.cat([x, skip_features], dim=1)
        # Apply the convolutional block.
        x = self.conv_relu(x)
        return x


class HybridGeoNet(nn.Module):
    # NOTE: Always initialize pre_decoder_proj in __init__ to avoid AttributeError.

    """
    The main hybrid model architecture for building detection/segmentation.
    It combines:
    1. A CNN Encoder (ResNet) for hierarchical feature extraction.
    2. A HybridViTGATBlock to process high-level CNN features for global and graph-based reasoning.
    3. A U-Net style Decoder for upsampling and producing the final segmentation mask,
       utilizing skip connections from the CNN encoder.
    """
    def __init__(self, num_input_channels=3, num_classes=1, # Default: 3 channels (RGB), 1 output class (binary segmentation)
                 cnn_model='resnet50', pretrained_cnn=True,
                 vit_embed_dim=768, vit_depth=4, vit_heads=12, gat_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.pre_decoder_proj = None  # Ensure projection attribute always exists

        # CNN Encoder: Extracts multi-scale features.
        self.cnn_encoder = CNNEncoder(cnn_model, num_input_channels, pretrained_cnn)
        # Get the number of channels from the highest-level CNN feature map (x5).
        cnn_high_level_channels = self.cnn_encoder.out_channels[3] # 2048 for ResNet50/101

        # Hybrid ViT-GAT Block: Processes the highest-level CNN features.
        # This block performs self-attention and graph attention.
        self.hybrid_vit_gat_block = HybridViTGATBlock(
            in_channels=cnn_high_level_channels, # Input channels from CNN (e.g., 2048)
            embed_dim=vit_embed_dim, # Embedding dimension for ViT attention (e.g., 768)
            num_heads=vit_heads, # Number of attention heads for ViT-like attention (e.g., 12)
            gat_heads=gat_heads # Number of attention heads for GAT (e.g., 4)
        )
        # The output channels of the HybridViTGATBlock is `embed_dim // gat_heads` (e.g., 192).
        decoder_initial_channels = vit_embed_dim // gat_heads

        # Decoder Path: U-Net style upsampling with skip connections.
        # First convolutional layer after the HybridViTGATBlock.
        self.first_decoder_conv = nn.Sequential(
            nn.Conv2d(decoder_initial_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Decoder blocks, each performing upsampling and concatenation with skip features.
        # `decoder_up4`: Upsamples from `512` channels, concatenates with `x4` (1024 channels), outputs `512`.
        self.decoder_up4 = DecoderBlock(512, self.cnn_encoder.out_channels[2], 512)
        # `decoder_up3`: Upsamples from `512` channels, concatenates with `x3` (512 channels), outputs `256`.
        self.decoder_up3 = DecoderBlock(512, self.cnn_encoder.out_channels[1], 256)
        # `decoder_up2`: Upsamples from `256` channels, concatenates with `x2` (256 channels), outputs `128`.
        self.decoder_up2 = DecoderBlock(256, self.cnn_encoder.out_channels[0], 128)

        # Final upsampling and convolutional layers to reach the original input image resolution.
        # Upsamples from `128` channels to `64`.
        self.final_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Final convolutional block before the output layer.
        self.final_conv_relu = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output convolution: Projects features to the number of output classes (e.g., 1 for binary segmentation).
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        """
        x : [B, C_in, H, W]
        returns output_mask : [B, num_classes, H, W]
        """
        B, _, H_in, W_in = x.shape

        # ---------- Encoder ----------
        x2, x3, x4, x5 = self.cnn_encoder(x)

        # ---------- ViT‑GAT ----------
        x = self.hybrid_vit_gat_block(x5)          # [B,192,H/32,W/32] or [B,96,H/32,W/32] (actual)

        # --- Channel Mismatch Fix ---
        # IMPORTANT: The decoder expects input with 192 channels here (first_decoder_conv expects 192 in-channels),
        # but the output of the encoder/ViT-GAT block may change depending on model complexity, number of heads, or MLP width.
        # If you change the ViT-GAT block's output channels (e.g., by altering number of GAT heads, embed_dim, or MLP structure),
        # you MUST ensure the output here matches the decoder's expected input channels.
        # This 1x1 Conv2d projects the encoder output to 192 channels if needed, preventing runtime errors.
        #
        # WHEN FINE-TUNING: Always check the shape here after any architectural change upstream!
        # If you see a RuntimeError about channel mismatch ("expected input[..., 192, ...], but got ..."),
        # this is the place to fix it. Adjust this projection or update the decoder as needed.
        if x.shape[1] != 192:
            # Persistent projection layer: Used only if the ViT-GAT output channels do not match decoder's expected channels (192)
            if self.pre_decoder_proj is None or self.pre_decoder_proj.in_channels != x.shape[1]:
                self.pre_decoder_proj = nn.Conv2d(x.shape[1], 192, kernel_size=1).to(x.device)
            x = self.pre_decoder_proj(x)
            print(f"[Fix] Projected encoder output to 192 channels: {x.shape}")

        # ---------- Decoder ----------
        x = self.first_decoder_conv(x)             # [B,512,H/32,W/32]
        x = self.decoder_up4(x, x4)                # [B,512,H/16,W/16]
        x = self.decoder_up3(x, x3)                # [B,256,H/8 ,W/8 ]
        x = self.decoder_up2(x, x2)                # [B,128,H/4 ,W/4 ]

        # ↓↓↓ *** critical: 128 → 64 before full‑res up‑scaling ***
        x = self.final_upsample(x)                 # [B,64 ,H/2 ,W/2]

        # up‑scale to original spatial size
        x = F.interpolate(x, size=(H_in, W_in),
                        mode="bilinear", align_corners=False)     # [B,64,H,W]

        # ---------- Output head ----------
        x = self.final_conv_relu(x)                # [B,32,H,W]
        output_mask = self.output_conv(x)          # [B,num_classes,H,W]

        return output_mask

