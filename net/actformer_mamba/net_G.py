import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import sys
import os
from net.actformer.transformer_utils import trunc_normal_, positional_encoding



from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class ActFormer_Generator(nn.Module):
    """
    ⁘ ∴ : ·
    Z: noise channel
    T: sequence length
    C: data channel
    V: data vertex
    spectral_norm: false* | true
    out_normalize: None* | skeleton | smpl

    mamba-related args: ...
    """

    def __init__(self,
                 Z=128,
                 T=64,
                 C=3,
                 V=25,
                 spectral_norm=True,
                 out_normalize=None,
                 
                learnable_pos_embed=True,
                embed_dim_ratio=16,
                depth=6,
                #  num_heads=8,
                #  mlp_ratio=2.0,
                #  qkv_bias=True,
                #  qk_scale=None,
                drop_rate=0.0,
                #  attn_drop_rate=0.0,
                drop_path_rate=0.2,
                #  norm_layer=None,
                
                num_class=60,
                ssm_cfg=None,
                norm_epsilon: float = 1e-5,
                rms_norm: bool = False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.Z = Z
        self.T = T
        self.C = C
        self.V = V
        self.spectral_norm = spectral_norm
        self.out_normalize = out_normalize
        embed_dim = embed_dim_ratio * V

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Frame-wise input embedding
        self.input_embedding = nn.Linear(Z, embed_dim)
        if spectral_norm:
            self.input_embedding = nn.utils.spectral_norm(self.input_embedding)

        # Class label condition embedding
        self.class_embedding = nn.Embedding(num_class, embed_dim)

        # Positional encoding
        if learnable_pos_embed:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, T+1, embed_dim))
            trunc_normal_(self.temporal_pos_embed, std=.02)
        else:
            temporal_pos_embed = positional_encoding(embed_dim, T)
            class_pos_embed = torch.zeros(1, embed_dim)
            self.temporal_pos_embed = nn.Parameter(torch.cat((class_pos_embed, temporal_pos_embed), 0).unsqueeze(0))
            self.temporal_pos_embed.requires_grad_(False)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
        #         spectral_norm=spectral_norm)
        #     for i in range(depth)])
        
        self.blocks = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.temporal_norm = nn.utils.spectral_norm(norm_layer(embed_dim)) if spectral_norm else norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, C*V)
        if spectral_norm:
            self.head = nn.utils.spectral_norm(self.head)
        self.tanh = nn.Tanh()

        nn.init.orthogonal_(self.class_embedding.weight)


    def forward(self, z, y,inference_params=None):
        if len(z.shape) == 2:
            z = z.unsqueeze(1).repeat(1, self.T, 1)
        else:
            z = z.squeeze(2).transpose(1, 2)

        # Input projection
        x = self.input_embedding(z)
        y = self.class_embedding(y).unsqueeze(1)
        x = torch.cat((y, x), 1)
        x += self.temporal_pos_embed

        # Transformer encoders
        x = self.pos_drop(x)
        residual = None
        for blk in self.blocks:
            x,residual= blk(x,residual,inference_params=inference_params)

        # Output projection
        x = self.temporal_norm(x)
        x = self.head(x)

        x = x[:, 1:].view(x.shape[0], self.T, self.V, -1)
        x = x.transpose(1, 3)

        # Normalize the output
        if self.out_normalize == 'skeleton':
            root, pose = x.split([1, self.V - 1], 2)
            pose = self.tanh(pose)
            pose = torch.split(pose, 3, 1)
            pose_p = []
            for p in range(len(pose)):
                n = pose[p].norm(dim=1, keepdim=True) + 1e-4
                pose_p.append(pose[p] / n)
            x = torch.cat((root, torch.cat(pose_p, 1)), 2)
        elif self.out_normalize == 'smpl':
            pose, root = x.split([self.V - 1, 1], 2)
            pose = self.tanh(pose)
            pose = torch.split(pose, 6, 1)
            pose_p = []
            for p in range(len(pose)):
                a1, a2 = pose[p][:, :3], pose[p][:, 3:]
                b1 = F.normalize(a1, dim=1)
                b2 = a2 - (b1 * a2).sum(1, keepdim=True) * b1
                b2 = F.normalize(b2, dim=1)
                pose_p.append(torch.cat((b1, b2), 1))
            x = torch.cat((torch.cat(pose_p, 1), root), 2)

        assert x.size()[1:-1] == (self.C, self.V)
        return x