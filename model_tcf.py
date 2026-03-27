import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math 

from model.Unet3D import Unet3D 

class PositionalEncoding(nn.Module):
    # Standard Sinusoidal Positional Encoding
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    # Standard Transformer Encoder Layer
    def __init__(self, num_heads, input_dim, dropout_rate=0.3, ffn_expansion_factor=4): 
        super(TransformerBlock, self).__init__()
        assert input_dim % num_heads == 0
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, 
                                                     batch_first=True, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * ffn_expansion_factor), 
            nn.GELU(),  
            nn.Dropout(dropout_rate), 
            nn.Linear(input_dim * ffn_expansion_factor, input_dim) 
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        x_norm2 = self.layer_norm2(x)
        ffn_output = self.ffn(x_norm2) 
        x = x + self.dropout2(ffn_output)
        return x

class CrossAttentionBlock(nn.Module):
    # Cross-modal Attention Module
    def __init__(self, num_heads, embed_dim, dropout_rate=0.3, ffn_expansion_factor=4):
        super(CrossAttentionBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm_kv = nn.LayerNorm(embed_dim)  
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                                batch_first=True, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * ffn_expansion_factor, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        query_norm = self.layer_norm1(query)
        key_norm = self.layer_norm_kv(key)
        value_norm = self.layer_norm_kv(value)
        attn_output, _ = self.cross_attn(query_norm, key_norm, value_norm) 
        query = query + self.dropout1(attn_output) 
        query = query + self.dropout2(self.ffn(self.layer_norm2(query)))
        return query 

class Module3D(nn.Module):
    # Feature extraction from 3D image sequences
    def __init__(self, obs_len, image_C, image_H, image_W, num_transformer_layers=3, dropout_rate=0.3,
                 unet_output_feature_dim=256): 
        super(Module3D, self).__init__()
        self.unet_feature_extractor = Unet3D(image_C, 1, obs_len=obs_len) 
        unet_reduced_dim = image_H * image_W 
        
        self.unet_output_projector = nn.Sequential(
            nn.Linear(unet_reduced_dim, unet_output_feature_dim),
            nn.LayerNorm(unet_output_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(unet_output_feature_dim, unet_output_feature_dim),
            nn.LayerNorm(unet_output_feature_dim)
        )
        
        self.positional_encoding = PositionalEncoding(unet_output_feature_dim, max_len=obs_len)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(num_heads=8, input_dim=unet_output_feature_dim, dropout_rate=dropout_rate)
              for _ in range(num_transformer_layers)]
        )

    def forward(self, x):
        unet_output = self.unet_feature_extractor(x) # (B, 1, T, H, W)
        batch_size, c, t, h, w = unet_output.size()
        # Flatten spatial dims and project to transformer space
        feat_flat = unet_output.permute(0, 2, 1, 3, 4).reshape(batch_size, t, c * h * w)
        feat_proj = self.unet_output_projector(feat_flat)
        x_transformed = self.transformer_layers(self.positional_encoding(feat_proj))
        return x_transformed

class ModuleEnv(nn.Module):
    # Processing environmental dictionary data
    def __init__(self, obs_len, env_feature_dims_dict, num_transformer_layers=2, dropout_rate=0.3):
        super(ModuleEnv, self).__init__()
        self.env_feature_mappers = nn.ModuleDict() 
        self.embedding_dim = 64  
        
        for key, dim in env_feature_dims_dict.items():
            self.env_feature_mappers[key] = nn.Sequential(
                nn.Linear(dim, self.embedding_dim * 2), 
                nn.LayerNorm(self.embedding_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate), 
                nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            )
        
        total_dim = len(env_feature_dims_dict) * self.embedding_dim
        self.positional_encoding = PositionalEncoding(total_dim, max_len=obs_len)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(num_heads=8, input_dim=total_dim, dropout_rate=dropout_rate)
              for _ in range(num_transformer_layers)]
        )

    def forward(self, env_data_dict):
        embedded_features = [self.env_feature_mappers[k](v) for k, v in env_data_dict.items()]
        combined_input = torch.cat(embedded_features, dim=2)
        x_transformed = self.transformer_layers(self.positional_encoding(combined_input))
        return x_transformed

class Module1D(nn.Module):
    # Processing trajectory, pressure center, date mask and tcf data
    def __init__(self, obs_len, traj_xy_dim, traj_Me_dim, date_mask_dim, tcf_data_dim=1, num_transformer_layers=3, dropout_rate=0.4):
        super(Module1D, self).__init__()
        original_input_dim = traj_xy_dim + traj_Me_dim + date_mask_dim + tcf_data_dim
        transformer_input_dim = 256 

        self.input_mapper = nn.Sequential(
            nn.Linear(original_input_dim, transformer_input_dim // 2),
            nn.LayerNorm(transformer_input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_input_dim // 2, transformer_input_dim),
            nn.LayerNorm(transformer_input_dim) 
        )
        
        self.positional_encoding = PositionalEncoding(transformer_input_dim, max_len=obs_len) 
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(num_heads=8, input_dim=transformer_input_dim, dropout_rate=dropout_rate)
              for _ in range(num_transformer_layers)]
        )

    def forward(self, obs_traj_rel, obs_traj_Me, obs_date_mask, obs_tcf_data):
        combined_1d = torch.cat([obs_traj_rel, obs_traj_Me, obs_date_mask, obs_tcf_data], dim=2)
        combined_1d = self.input_mapper(combined_1d)
        x_transformed = self.transformer_layers(self.positional_encoding(combined_1d))
        return x_transformed

class MultiScaleFusion(nn.Module):
    # Multi-scale temporal feature extraction via dilated convolutions
    def __init__(self, embed_dim, num_scales=3, dropout_rate=0.3):
        super().__init__()
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=4, dilation=4)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        x_t = x.transpose(1, 2)
        multi_scale_features = [F.gelu(conv(x_t)) for conv in self.scale_convs]
        concat_feat = torch.cat(multi_scale_features, dim=1).transpose(1, 2)
        return self.fusion(concat_feat) + x  

class TransformerDecoderHead(nn.Module):
    # Prediction head using Transformer Decoder
    def __init__(self, hidden_dim, pred_len, output_feature_dim, num_decoder_layers=3, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.pred_len = pred_len
        self.query_embed = nn.Embedding(pred_len, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=pred_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, 
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_feature_dim)
        )

    def forward(self, encoder_output_seq):
        batch_size = encoder_output_seq.size(0)
        positions = torch.arange(self.pred_len, device=encoder_output_seq.device)
        query = self.query_embed(positions).unsqueeze(0).repeat(batch_size, 1, 1) 
        decoder_output = self.transformer_decoder(tgt=self.positional_encoding(query), memory=encoder_output_seq)
        return self.output_proj(decoder_output)

class MyCombinedModel(nn.Module):
    # Main model for multi-modal feature fusion and prediction
    def __init__(self, obs_len, pred_len,
                 image_C, image_H, image_W,
                 env_feature_dims_dict,
                 traj_xy_dim=2, traj_Me_dim=2, date_mask_dim=4, tcf_data_dim=1,
                 hidden_dim=256, output_dim=2, 
                 num_3d_transformer_layers=3,
                 num_env_transformer_layers=3,
                 num_1d_transformer_layers=3,
                 num_cross_attention_heads=8,
                 num_post_cross_attn_self_attn_layers=4, 
                 num_decoder_layers_for_head=3,
                 unet_output_feature_dim=256, 
                 dropout_rate=0.2
                ):
        super(MyCombinedModel, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.fusion_embed_dim = hidden_dim 

        self.module3d = Module3D(obs_len, image_C, image_H, image_W, num_3d_transformer_layers, dropout_rate, unet_output_feature_dim)
        self.module_env = ModuleEnv(obs_len, env_feature_dims_dict, num_env_transformer_layers, dropout_rate)
        self.module1d = Module1D(obs_len, traj_xy_dim, traj_Me_dim, date_mask_dim, tcf_data_dim, num_1d_transformer_layers, 0.4)

        # Projection layers to align dimensions
        self.proj_3d = nn.Sequential(nn.Linear(unet_output_feature_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.proj_env = nn.Sequential(nn.Linear(len(env_feature_dims_dict) * 64, hidden_dim), nn.LayerNorm(hidden_dim))
        self.proj_1d = nn.Sequential(nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim))
        self.multi_scale_fusion = MultiScaleFusion(hidden_dim, num_scales=3, dropout_rate=dropout_rate)

        # Modality-specific learnable embeddings
        self.modality_emb_3d = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_emb_env = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_emb_1d = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.symmetric_fusion_layers = nn.Sequential(
            *[TransformerBlock(num_heads=num_cross_attention_heads, input_dim=hidden_dim, dropout_rate=dropout_rate)
              for _ in range(num_post_cross_attn_self_attn_layers)]
        )
        
        self.intensity_fusion_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.intensity_env_bridge = nn.Linear(hidden_dim, hidden_dim)

        self.intensity_head = TransformerDecoderHead(hidden_dim=hidden_dim, pred_len=pred_len, output_feature_dim=output_dim, 
                                                     num_decoder_layers=num_decoder_layers_for_head, num_heads=num_cross_attention_heads, dropout_rate=dropout_rate)

    def forward(self, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
                loss_mask, seq_start_end, obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
                obs_date_mask, pred_date_mask, obs_tcf_data, pred_tcf_data,
                image_obs, image_pre, env_data_dict, tyID,
                num_samples=6):
        
        # Extract features from each module
        f_3d = self.proj_3d(self.module3d(image_obs)) + self.modality_emb_3d
        f_env = self.proj_env(self.module_env(env_data_dict)) + self.modality_emb_env
        f_1d = self.multi_scale_fusion(self.proj_1d(self.module1d(obs_traj_rel, obs_traj_Me, obs_date_mask, obs_tcf_data))) + self.modality_emb_1d

        # Fuse modalities using transformer layers
        fused_sequence = self.symmetric_fusion_layers(torch.cat([f_3d, f_env, f_1d], dim=1))
        
        # Process fused sequence for intensity prediction
        intensity_features = self.intensity_fusion_processor(fused_sequence) + self.intensity_env_bridge(fused_sequence)
        intensity_predictions = self.intensity_head(intensity_features)

        # Final predictions formatted for multiple samples
        predictions = intensity_predictions.unsqueeze(1) 
        if num_samples > 1:
            predictions = predictions.repeat(1, num_samples, 1, 1)
        
        return predictions

