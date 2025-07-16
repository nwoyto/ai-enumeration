import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
# Import GATConv and Data from torch_geometric (ensure installation)
try:
    from torch_geometric.nn import GATConv
    # from torch_geometric.data import Data # Data class is for graph data structure
except ImportError:
    print("WARNING: torch_geometric not found. GAT will use dummy implementation.")
    # Define dummy classes if PyG is not installed, to allow basic testing

class SpaceNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', num_input_channels=3, pretrained=True):
        super().__init__()
        # ... (CNN backbone initialization as discussed)
        self.backbone = models.resnet50(pretrained=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            num_input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        if num_input_channels == 3 and pretrained:
             self.backbone = models.resnet50(pretrained=True)
        elif pretrained:
             pretrained_state_dict = models.resnet50(pretrained=True).state_dict()
             new_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'conv1' not in k}
             self.backbone.load_state_dict(new_state_dict, strict=False)

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.feature_dim = 2048

    def forward(self, x):
        return self.backbone(x)

class HybridBuildingDetector(nn.Module):
    def __init__(self, cnn_model='resnet50', num_input_channels=3,
                 vit_patch_size=16, vit_embed_dim=768, vit_depth=6, vit_heads=12,
                 gat_hidden_channels=256, gat_heads=4, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # 1. CNN Feature Extractor
        self.cnn_extractor = SpaceNetFeatureExtractor(cnn_model, num_input_channels)
        cnn_out_channels = self.cnn_extractor.feature_dim
        cnn_feature_map_size = 7 # Assuming 224x224 input and ResNet50

        # 2. Visual Transformer (ViT) Layer
        self.vit_model = VisionTransformer(
            img_size=cnn_feature_map_size,
            patch_size=1, # Each CNN feature cell is a token
            in_chans=cnn_out_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            global_pool='' # No global pooling or class token for spatial output
        )

        # 3. Graph Attention Mechanism (on ViT output tokens)
        self.graph_attention_channels = gat_hidden_channels
        self.gat_proc_linear = nn.Linear(vit_embed_dim, self.graph_attention_channels)
        self.gat_conv1 = GATConv(self.graph_attention_channels, self.graph_attention_channels, heads=gat_heads, dropout=0.6)
        self.gat_conv2 = GATConv(self.graph_attention_channels * gat_heads, self.graph_attention_channels, heads=1, concat=False, dropout=0.6)
        self.dropout = nn.Dropout(0.6)


        # 4. Segmentation Head (Decoder to upsample features to mask)
        # This part combines CNN_features and GAT_features
        combined_feature_channels = cnn_out_channels + self.graph_attention_channels # Output channels from GAT (gat_hidden_channels)
        
        self.final_upsample_conv = nn.Conv2d(combined_feature_channels, 256, kernel_size=3, padding=1)
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            ),
             nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            )
        ])
        self.output_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        B, _, H_img, W_img = x.shape

        # 1. CNN Features
        cnn_features = self.cnn_extractor(x)
        _, C_feat, H_feat, W_feat = cnn_features.shape

        # 2. ViT Processing
        vit_output_tokens = self.vit_model(cnn_features) # (B, H_feat*W_feat, embed_dim) if global_pool='', else (B, tokens+1, embed_dim)
        
        # If the ViT uses a class token (i.e., output is B x (num_patches+1) x embed_dim), remove it:
        # Check if vit_output_tokens has a class token (e.g., if timm ViT default includes one)
        # Assuming num_cnn_patches is H_feat * W_feat (7*7=49)
        num_cnn_patches = H_feat * W_feat
        if vit_output_tokens.dim() == 3 and vit_output_tokens.shape[1] == num_cnn_patches + 1:
             vit_output_tokens = vit_output_tokens[:, 1:, :] # Remove class token

        # Reshape ViT output tokens to spatial feature map (B, embed_dim, H_feat, W_feat)
        vit_spatial_features = vit_output_tokens.permute(0, 2, 1).reshape(B, self.vit_model.embed_dim, H_feat, W_feat)

        # 3. Graph Attention Mechanism
        # Convert ViT spatial features to node features for GAT
        node_features_flat = vit_spatial_features.permute(0, 2, 3, 1).reshape(-1, self.vit_model.embed_dim) # (B*H*W, E)
        node_features_gat_input = self.gat_proc_linear(node_features_flat)
        node_features_gat_input = self.dropout(node_features_gat_input)

        # Create batch-wise grid-like edge index for GAT (simplified for demonstration)
        # This part assumes a regular grid graph for each image in the batch
        edge_index_list = []
        batch_indices = []
        for i in range(B):
            num_nodes_per_batch = H_feat * W_feat
            node_offset = i * num_nodes_per_batch
            # Add horizontal edges
            for r in range(H_feat):
                for c in range(W_feat - 1):
                    idx1 = r * W_feat + c + node_offset
                    idx2 = r * W_feat + (c + 1) + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            # Add vertical edges
            for r in range(H_feat - 1):
                for c in range(W_feat):
                    idx1 = r * W_feat + c + node_offset
                    idx2 = (r + 1) * W_feat + c + node_offset
                    edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            batch_indices.extend([i] * num_nodes_per_batch)
        
        # Convert to PyG's expected format
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(x.device)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(x.device)

        x_gat = self.gat_conv1(node_features_gat_input, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = self.dropout(x_gat)
        x_gat = self.gat_conv2(x_gat, edge_index)
        
        # Reshape GAT output back to spatial feature map
        gat_spatial_features = x_gat.reshape(B, H_feat, W_feat, self.graph_attention_channels).permute(0, 3, 1, 2)

        # 4. Concatenate CNN features and GAT-enhanced features
        combined_features = torch.cat((cnn_features, gat_spatial_features), dim=1)

        # 5. Segmentation Head (Decoder)
        x = self.final_upsample_conv(combined_features)
        x = F.relu(x, inplace=True)

        for block in self.upsample_blocks:
            x = block(x)

        output_mask = self.output_conv(x) # Final segmentation logits

        return output_mask