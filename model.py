import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math 
from TCNM.Unet3D_merge_tiny2 import Unet3D 


class PositionalEncoding(nn.Module):
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
    def __init__(self, num_heads, input_dim, dropout_rate=0.3, ffn_expansion_factor=4): 
        super(TransformerBlock, self).__init__()
        assert input_dim % num_heads == 0, f"embed_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
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
    def __init__(self, num_heads, embed_dim, dropout_rate=0.3, ffn_expansion_factor=4):
        super(CrossAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
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
        query_norm2 = self.layer_norm2(query)
        ffn_output = self.ffn(query_norm2)
        query = query + self.dropout2(ffn_output)
        return query 

# --- Module3D ---
class Module3D(nn.Module):
    def __init__(self, obs_len, image_C, image_H, image_W, num_transformer_layers=3, dropout_rate=0.3,
                 unet_output_feature_dim=256): 
        super(Module3D, self).__init__()
        self.obs_len = obs_len 
        self.image_C = image_C 
        self.image_H = image_H
        self.image_W = image_W
        self.unet_feature_extractor = Unet3D(self.image_C, 1, obs_len=obs_len) 
        unet_reduced_dim = image_H * image_W 
        self.unet_output_projector = nn.Sequential(
            nn.Linear(unet_reduced_dim, unet_output_feature_dim),
            nn.LayerNorm(unet_output_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(unet_output_feature_dim, unet_output_feature_dim),
            nn.LayerNorm(unet_output_feature_dim)
        )
        transformer_input_dim = unet_output_feature_dim 
        self.positional_encoding = PositionalEncoding(transformer_input_dim, max_len=obs_len)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(num_heads=8, input_dim=transformer_input_dim, dropout_rate=dropout_rate)
              for _ in range(num_transformer_layers)]
        )

    def forward(self, x):
        unet_output = self.unet_feature_extractor(x)
        batch_size, c, t, h, w = unet_output.size()
        frame_features_flattened = unet_output.permute(0, 2, 1, 3, 4).reshape(batch_size, t, c * h * w)
        frame_features_projected = self.unet_output_projector(frame_features_flattened)
        frame_features_projected = self.positional_encoding(frame_features_projected)
        x_transformed = self.transformer_layers(frame_features_projected)
        return x_transformed

# --- ModuleEnv ---
class ModuleEnv(nn.Module):
    def __init__(self, obs_len, env_feature_dims_dict, num_transformer_layers=2, dropout_rate=0.1):
        super(ModuleEnv, self).__init__()
        self.obs_len = obs_len
        self.env_feature_dims_dict = env_feature_dims_dict
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
        total_input_dim_per_step = len(env_feature_dims_dict) * self.embedding_dim
        self.transformer_num_heads = 8
        self.positional_encoding = PositionalEncoding(total_input_dim_per_step, max_len=obs_len)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(num_heads=self.transformer_num_heads, input_dim=total_input_dim_per_step, dropout_rate=dropout_rate)
              for _ in range(num_transformer_layers)]
        )

    def forward(self, env_data_dict):
        embedded_features_list_per_step = []
        for key in self.env_feature_mappers:
            x = env_data_dict[key]
            x_mapped = self.env_feature_mappers[key](x)
            embedded_features_list_per_step.append(x_mapped)
        combined_env_input = torch.cat(embedded_features_list_per_step, dim=2)
        combined_env_input = self.positional_encoding(combined_env_input)
        x_transformed = self.transformer_layers(combined_env_input)
        return x_transformed

# --- Module1D ---
class Module1D(nn.Module):
    def __init__(self, obs_len, traj_xy_dim, traj_Me_dim, date_mask_dim, extra_data_dim=1, num_transformer_layers=3, dropout_rate=0.1, use_extra_data=True):
        super(Module1D, self).__init__()
        self.obs_len = obs_len
        self.use_extra_data = use_extra_data
        actual_extra_dim = extra_data_dim if use_extra_data else 0
        original_input_dim_per_step = traj_xy_dim + traj_Me_dim + date_mask_dim + actual_extra_dim
        transformer_input_dim = 256 
        self.input_mapper = nn.Sequential(
            nn.Linear(original_input_dim_per_step, transformer_input_dim // 2),
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

    def forward(self, obs_traj_rel, obs_traj_Me, obs_date_mask, obs_extra_data):
        input_list = [obs_traj_rel, obs_traj_Me, obs_date_mask]
        if self.use_extra_data:
            input_list.append(obs_extra_data)
        combined_1d_input_original_dim = torch.cat(input_list, dim=2)
        combined_1d_input = self.input_mapper(combined_1d_input_original_dim)
        combined_1d_input = self.positional_encoding(combined_1d_input)
        x_transformed = self.transformer_layers(combined_1d_input)
        return x_transformed

# --- MultiScaleFusion ---
class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim, num_scales=3, dropout_rate=0.3):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
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
        multi_scale_features = []
        for conv in self.scale_convs:
            feat = F.gelu(conv(x_t))
            multi_scale_features.append(feat)
        concat_feat = torch.cat(multi_scale_features, dim=1).transpose(1, 2)
        fused = self.fusion(concat_feat)
        return fused + x

# --- TransformerDecoderHead ---
class TransformerDecoderHead(nn.Module):
    def __init__(self, hidden_dim, pred_len, output_feature_dim, num_decoder_layers=3, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.query_embed = nn.Embedding(pred_len, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=pred_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout_rate, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_feature_dim)
        )

    def forward(self, encoder_output_seq):
        batch_size = encoder_output_seq.size(0)
        positions = torch.arange(self.pred_len, device=encoder_output_seq.device)
        query = self.query_embed(positions).unsqueeze(0).repeat(batch_size, 1, 1) 
        query = self.positional_encoding(query)
        decoder_output = self.transformer_decoder(tgt=query, memory=encoder_output_seq)
        return self.output_proj(decoder_output)


class GCBlock(nn.Module):
    def __init__(self, dim, ratio=0.25):
        super(GCBlock, self).__init__()
        self.dim = dim
        # 全局上下文建模：学习序列中各步的权重
        self.query_conv = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # 特征转换（Bottleneck 结构）
        self.channel_add_conv = nn.Sequential(
            nn.Linear(dim, int(dim * ratio)),
            nn.LayerNorm(int(dim * ratio)),
            nn.GELU(),
            nn.Linear(int(dim * ratio), dim)
        )

    def forward(self, x):
        # x shape: (B, L, C)
        b, l, c = x.size()
        input_x = x
        
        # 1. Context Modeling: (B, L, 1) -> (B, 1, L)
        context_mask = self.query_conv(x).transpose(1, 2)
        context_mask = self.softmax(context_mask)
        
        # 2. Global Pooling: (B, 1, L) @ (B, L, C) -> (B, 1, C)
        context = torch.matmul(context_mask, input_x)
        
        # 3. Transform & Fusion
        channel_add_term = self.channel_add_conv(context)
        return x + channel_add_term

class MyCombinedModel(nn.Module):
    def __init__(self, obs_len, pred_len,
                 image_C, image_H, image_W,
                 env_feature_dims_dict,
                 traj_xy_dim=2, traj_Me_dim=2, date_mask_dim=4, extra_data_dim=1,
                 hidden_dim=256, output_dim=2, 
                 num_3d_transformer_layers=3,
                 num_env_transformer_layers=3,
                 num_1d_transformer_layers=3,
                 num_cross_attention_heads=8,
                 num_cross_attention_layers=3,
                 num_post_cross_attn_self_attn_layers=4, 
                 num_decoder_layers_for_head=3,
                 unet_output_feature_dim=256, 
                 dropout_rate=0.3,
                 use_extra_data=True,
                 num_gs=6  
                ):
        super(MyCombinedModel, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.output_dim = output_dim 
        self.fusion_embed_dim = hidden_dim 
        self.use_extra_data = use_extra_data
        self.num_gs = num_gs

        self.module3d = Module3D(obs_len, image_C, image_H, image_W, 
                                 num_transformer_layers=num_3d_transformer_layers, 
                                 dropout_rate=dropout_rate, 
                                 unet_output_feature_dim=unet_output_feature_dim)
        
        self.module_env = ModuleEnv(obs_len, env_feature_dims_dict, 
                                  num_transformer_layers=num_env_transformer_layers, 
                                  dropout_rate=dropout_rate)
        
        self.module1d = Module1D(obs_len, traj_xy_dim, traj_Me_dim, date_mask_dim, extra_data_dim,
                                 num_transformer_layers=num_1d_transformer_layers, 
                                 dropout_rate=0.4, use_extra_data=use_extra_data)


        self.proj_3d = nn.Sequential(nn.Linear(unet_output_feature_dim, self.fusion_embed_dim), nn.LayerNorm(self.fusion_embed_dim))
        self.proj_env = nn.Sequential(nn.Linear(len(env_feature_dims_dict) * 64, self.fusion_embed_dim), nn.LayerNorm(self.fusion_embed_dim))
        self.proj_1d = nn.Sequential(nn.Linear(256, self.fusion_embed_dim), nn.LayerNorm(self.fusion_embed_dim))
        
        self.multi_scale_fusion = MultiScaleFusion(self.fusion_embed_dim, num_scales=3, dropout_rate=dropout_rate)

        self.modality_emb_3d = nn.Parameter(torch.randn(1, 1, self.fusion_embed_dim))
        self.modality_emb_env = nn.Parameter(torch.randn(1, 1, self.fusion_embed_dim))
        self.modality_emb_1d = nn.Parameter(torch.randn(1, 1, self.fusion_embed_dim))

       
        self.symmetric_fusion_layers = nn.Sequential(
            *[TransformerBlock(num_heads=num_cross_attention_heads, input_dim=self.fusion_embed_dim, dropout_rate=dropout_rate)
              for _ in range(num_post_cross_attn_self_attn_layers)]
        )
        
    
        self.gc_net = GCBlock(dim=self.fusion_embed_dim)

        self.net_chooser = nn.Sequential(
            nn.Linear(self.fusion_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_gs),
        )

       
        self.intensity_heads = nn.ModuleList([
            TransformerDecoderHead(hidden_dim=hidden_dim, 
                                   pred_len=pred_len, 
                                   output_feature_dim=output_dim, 
                                   num_decoder_layers=num_decoder_layers_for_head, 
                                   num_heads=num_cross_attention_heads, 
                                   dropout_rate=dropout_rate)
            for _ in range(num_gs)
        ])

        
        self.intensity_fusion_processor = nn.Sequential(
            nn.Linear(self.fusion_embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.intensity_env_bridge = nn.Linear(self.fusion_embed_dim, hidden_dim)

    def forward(self, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
                loss_mask, seq_start_end, obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
                obs_date_mask, pred_date_mask, obs_extra_data, pred_extra_data,
                image_obs, image_pre, env_data_dict, tyID,
                num_samples=6):
        
       
        seq_features_3d = self.module3d(image_obs) 
        seq_features_env = self.module_env(env_data_dict) 
        seq_features_1d = self.module1d(obs_traj_rel, obs_traj_Me, obs_date_mask, obs_extra_data) 
        
       
        f_3d = self.proj_3d(seq_features_3d) + self.modality_emb_3d
        f_env = self.proj_env(seq_features_env) + self.modality_emb_env
        f_1d = self.proj_1d(seq_features_1d) + self.modality_emb_1d
        f_1d = self.multi_scale_fusion(f_1d)

       
        combined_features = torch.cat([f_3d, f_env, f_1d], dim=1)
        fused_sequence = self.symmetric_fusion_layers(combined_features)
        
        # --- GC-Net ---
        fused_sequence = self.gc_net(fused_sequence)
        
        # 4. Net Chooser 
        global_feat = torch.mean(fused_sequence, dim=1) 
        net_chooser_out = self.net_chooser(global_feat) # (B, num_gs)

        intensity_base = self.intensity_fusion_processor(fused_sequence)
        intensity_features = intensity_base + self.intensity_env_bridge(fused_sequence)
        
        all_gs_preds = []
        for i in range(self.num_gs):
            
            pred = self.intensity_heads[i](intensity_features)
            all_gs_preds.append(pred.unsqueeze(1)) # (B, 1, pred_len, output_dim)

        # out put: (B, num_gs, pred_len, output_dim)
        predictions = torch.cat(all_gs_preds, dim=1)
        
        return predictions, net_chooser_out


