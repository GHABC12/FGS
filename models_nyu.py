#1
# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# --------------------------------------------------------

from functools import partial, reduce
from operator import mul
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint # 修改：使用原生 checkpoint
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp

from util.vitdet_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    LayerNorm2D,
)

# 启用 TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False,
            rel_pos_zero_init=True, input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, num_prompts=0, spatial_size=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_bias = None
        if self.use_rel_pos:
            if spatial_size is None:
                H, W = self.input_size
            else:
                H, W = spatial_size

            q_flat = q.reshape(B * self.num_heads, L, -1)
            q_img_flat = q_flat[:, num_prompts:, :]

            if num_prompts == 0:
                dummy_attn = torch.zeros(B * self.num_heads, L, L, device=q.device, dtype=q.dtype)
                bias_flat = add_decomposed_rel_pos(dummy_attn, q_flat, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                attn_bias = bias_flat.view(B, self.num_heads, L, L)
            else:
                dummy_attn_img = torch.zeros(B * self.num_heads, H * W, H * W, device=q.device, dtype=q.dtype)
                rel_pos_bias = add_decomposed_rel_pos(dummy_attn_img, q_img_flat, self.rel_pos_h, self.rel_pos_w,
                                                      (H, W), (H, W))
                padded_bias = F.pad(rel_pos_bias, (num_prompts, 0, num_prompts, 0))
                attn_bias = padded_bias.view(B, self.num_heads, L, L)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

class ResBottleneckBlock(CNNBlockBase):
    def __init__(self, in_channels, out_channels, bottleneck_channels, norm="LN", act_layer=nn.GELU):
        super().__init__(in_channels, out_channels, 1)
        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()
        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)
        out = x + out
        return out


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
            norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_rel_pos=False,
            rel_pos_zero_init=True, window_size=0, use_residual_block=False, input_size=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.window_size = window_size
        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(
                in_channels=dim, out_channels=dim, bottleneck_channels=dim // 2,
                norm="LN", act_layer=act_layer,
            )

    def forward(self, x, prompt_emb=None):
        B_in, L_in, C_in = x.shape
        H, W = self.input_size
        img_len = H * W
        num_prompts = L_in - img_len

        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            if num_prompts > 0:
                x_prompts = x[:, :num_prompts, :]
                x_img = x[:, num_prompts:, :]
            else:
                x_prompts = None
                x_img = x

            x_img = x_img.view(B_in, H, W, C_in)
            x_img_partitioned, pad_hw = window_partition(x_img, self.window_size)

            if x_prompts is not None:
                num_windows = x_img_partitioned.shape[0] // B_in
                x_prompts_expanded = x_prompts.unsqueeze(1).expand(-1, num_windows, -1, -1).reshape(-1, num_prompts, C_in)
                x_input = torch.cat([x_prompts_expanded, x_img_partitioned], dim=1)
            else:
                x_input = x_img_partitioned

            spatial_size = (self.window_size, self.window_size)
            x_attn = self.attn(x_input, num_prompts=num_prompts, spatial_size=spatial_size)

            if num_prompts > 0:
                out_prompts = x_attn[:, :num_prompts, :]
                out_img = x_attn[:, num_prompts:, :]
                out_prompts = out_prompts.view(B_in, -1, num_prompts, C_in).mean(dim=1)
            else:
                out_prompts = None
                out_img = x_attn

            out_img = out_img.view(-1, self.window_size, self.window_size, C_in)
            out_img = window_unpartition(out_img, self.window_size, pad_hw, (H, W))
            out_img = out_img.view(B_in, H * W, C_in)

            if out_prompts is not None:
                x = torch.cat([out_prompts, out_img], dim=1)
            else:
                x = out_img
        else:
            spatial_size = (H, W)
            x = self.attn(x, num_prompts=num_prompts, spatial_size=spatial_size)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            if num_prompts > 0:
                final_prompts = x[:, :num_prompts, :]
                final_img = x[:, num_prompts:, :]
            else:
                final_prompts = None
                final_img = x

            res_part = final_img.view(B_in, H, W, C_in)
            res_part = self.residual(res_part.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten(1, 2)
            final_img = res_part

            if final_prompts is not None:
                x = torch.cat([final_prompts, final_img], dim=1)
            else:
                x = final_img

        return x


class Painter(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
            num_heads=16, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.,
            norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_abs_pos=True,
            use_rel_pos=False, rel_pos_zero_init=True, window_size=0,
            window_block_indexes=(), residual_block_indexes=(), use_act_checkpoint=False,
            pretrain_img_size=224, pretrain_use_cls_token=True, out_feature="last_feat",
            decoder_embed_dim=128, loss_func="smoothl1", vpt_on=False, prompt_length=100,
            vpt_type='shallow', vpt_target_layer=0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.vpt_on = vpt_on
        self.vpt_type = vpt_type
        self.prompt_length = prompt_length
        self.use_act_checkpoint = use_act_checkpoint

        if isinstance(vpt_target_layer, list):
            self.vpt_target_layers = sorted(vpt_target_layer)
        else:
            self.vpt_target_layers = [vpt_target_layer]

        if self.vpt_on:
            self.prompt_embeddings = nn.ParameterList([
                nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
                for _ in range(len(self.vpt_target_layers))
            ])
            for p in self.prompt_embeddings:
                nn.init.normal_(p.data, std=1e-5)
            self.prompt_dropout = nn.Dropout(0.1)

        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            # 修改：移除这里多余的 checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim * 4, patch_size ** 2 * self.decoder_embed_dim, bias=True)
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
            LayerNorm2D(self.decoder_embed_dim), nn.GELU(),
            nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True),
        )
        self.loss_func = loss_func

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0
        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        w = int((x.shape[1] * 0.5) ** .5)
        h = w * 2
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_decoder(self, x):
        x = torch.cat(x, dim=-1)
        x = self.decoder_embed(x)
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))
        x = self.decoder_pred(x)
        return x

    def forward_loss(self, pred, tgts, mask, valid):
        mask = mask[:, :, None].repeat(1, 1, self.patch_size ** 2 * 3)
        mask = self.unpatchify(mask)

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(tgts.device)[None, :, None, None]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(tgts.device)[None, :, None, None]
        inds_ign = ((tgts * imagenet_std + imagenet_mean) * (1 - 1. * mask)).sum((1, 2, 3)) < 100 * 3
        if inds_ign.sum() > 0:
            valid[inds_ign] = 0.
        if valid is None:
            valid = torch.ones_like(tgts)
        mask = mask * valid

        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss = (loss * mask).sum() / (mask.sum() + 1e-2)
        return loss

    def forward(self, x, tgt, mask, valid):
        tgt_original = tgt

        # 修改：原生 checkpoint 包装
        if self.use_act_checkpoint:
            x = checkpoint.checkpoint(self.patch_embed, x, use_reentrant=False)
            tgt = checkpoint.checkpoint(self.patch_embed, tgt, use_reentrant=False)
        else:
            x = self.patch_embed(x)
            tgt = self.patch_embed(tgt)

        batch_size, Hp, Wp, _ = x.size()
        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)
        w = mask.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        tgt = tgt * (1 - w) + mask_token * w

        x = x + self.segment_token_x
        tgt = tgt + self.segment_token_y

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))
            tgt = tgt + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (tgt.shape[1], tgt.shape[2]))

        x = torch.cat((x, tgt), dim=0)

        if x.dim() == 4:
            x = x.flatten(1, 2)

        merge_idx = 2
        out_feats = []
        img_len = self.patch_embed.num_patches

        intermediate_feats = {}
        anchor_layers = [0, 1, 2, 10, 11, 12, 21, 22, 23]

        for i, blk in enumerate(self.blocks):
            active_layers = getattr(self, 'active_vpt_layers', self.vpt_target_layers)
            if self.vpt_on and i in active_layers:
                idx_in_emb = self.vpt_target_layers.index(i)
                current_bs = x.shape[0]
                new_prompts = self.prompt_dropout(
                    self.prompt_embeddings[idx_in_emb].expand(current_bs, -1, -1)
                )

                current_len = x.shape[1]
                if current_len > img_len:
                    x_img = x[:, -img_len:, :]
                else:
                    x_img = x
                x = torch.cat([new_prompts, x_img], dim=1)

            # 修改：原生 checkpoint 包装
            if self.use_act_checkpoint and i != len(self.blocks) - 1:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

            if i == merge_idx:
                current_len = x.shape[1]
                num_prompts = current_len - img_len

                if num_prompts > 0:
                    x_prompts = x[:, :num_prompts, :]
                    x_img = x[:, num_prompts:, :]
                else:
                    x_prompts = None
                    x_img = x

                x_img = x_img.reshape(2, -1, x_img.shape[1], x_img.shape[2]).mean(dim=0)

                if x_prompts is not None:
                    x_prompts = x_prompts.reshape(2, -1, x_prompts.shape[1], x_prompts.shape[2]).mean(dim=0)
                    x = torch.cat([x_prompts, x_img], dim=1)
                else:
                    x = x_img

            if i in anchor_layers:
                current_len = x.shape[1]
                if current_len > img_len:
                    x_img_clean = x[:, -img_len:, :]
                else:
                    x_img_clean = x
                intermediate_feats[i] = x_img_clean.clone()

            if i in [5, 11, 17, 23]:
                current_len = x.shape[1]
                if current_len > img_len:
                    feat = x[:, -img_len:, :]
                else:
                    feat = x

                h_feat = self.img_size[0] // 16
                w_feat = self.img_size[1] // 16
                out_feats.append(self.norm(feat).view(feat.shape[0], h_feat, w_feat, -1))

        pred = self.forward_decoder(out_feats)
        loss = self.forward_loss(pred, tgt_original, mask, valid)

        return loss, self.patchify(pred), mask, intermediate_feats


def painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1(**kwargs):
    if 'use_act_checkpoint' not in kwargs:
        kwargs['use_act_checkpoint'] = False

    model = Painter(
        img_size=(896, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                              list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)