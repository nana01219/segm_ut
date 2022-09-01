"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, with_ut = False):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        if with_ut:
            self.with_ut = True
            self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        else:
            self.with_ut = False


    @property
    def unwrapped(self):
        return self

    def norm_uncertainty(self, uncertainty):
        # uncertainty = torch.log(torch.exp(uncertainty))
        uncertainty = (torch.tanh(uncertainty) + 1)/2
        return uncertainty

    def forward(self, x, mask=None, use_gate = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        if self.with_ut:
            qk = (q @ k.transpose(-2, -1)) 
            uncertainty = self.data_uncertainty(qk)
            uncertainty = self.norm_uncertainty(uncertainty)
            a, b, c, d = uncertainty.shape
            r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = (r>uncertainty)

            attn = qk * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn*mask

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn

class Attention_data(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num, act = "sigmoid"):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        # elif act == "logexp":

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

    def norm_uncertainty(self, uncertainty):
        # uncertainty = torch.log(torch.exp(uncertainty))
        # uncertainty = (torch.tanh(uncertainty) + 1)/2
        uncertainty = self.act(uncertainty)
        return uncertainty

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, use_gate = False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        
        qk = (q @ k.transpose(-2, -1)) 

        if self.repeat_num is not None:
            attn_list = []
            x_list = []
            uncertainty = self.data_uncertainty(qk)
            uncertainty = self.norm_uncertainty(uncertainty)
            a, b, c, d = uncertainty.shape

            r = torch.rand([self.repeat_num, a, b, c, d]).to(uncertainty)

            for i in range(self.repeat_num):
                mask = (r[i]>uncertainty)

                attn = qk * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                attn = attn*mask

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                attn_list.append(attn)
                x_list.append(x)

            return x_list, attn_list


        else:
            attn = qk * self.scale
            attn_mean = attn.softmax(dim=-1)

            uncertainty = self.data_uncertainty(qk)   # -inf, inf
            uncertainty = self.norm_uncertainty(uncertainty)  # 0, 1

            if use_gate:
                a, b, c, d = uncertainty.shape
                r = torch.rand([a, b, c, d]).to(uncertainty)
                mask = (r>uncertainty)
                attn = attn_mean*mask
            else:
                # attn = attn*uncertainty
                r = torch.randn_like(uncertainty).to(uncertainty)
                attn = attn_mean + uncertainty*r

            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn_mean, uncertainty

class Attention_relu(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num, act = "sigmoid"):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        # self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        # elif act == "logexp":

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

    def norm_uncertainty(self, uncertainty):
        # uncertainty = torch.log(torch.exp(uncertainty))
        # uncertainty = (torch.tanh(uncertainty) + 1)/2
        uncertainty = self.act(uncertainty)
        return uncertainty

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, use_gate = False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        
        qk = (q @ k.transpose(-2, -1)) 

        attn = qk * self.scale
        attn_mean = attn.softmax(dim=-1)

        # uncertainty = self.data_uncertainty(qk)   # -inf, inf
        uncertainty = self.norm_uncertainty(qk)  # 0, 1

        if True:
            a, b, c, d = uncertainty.shape
            r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = (r>uncertainty)
            attn = attn_mean*mask
        else:
            # attn = attn*uncertainty
            r = torch.randn_like(uncertainty).to(uncertainty)
            attn = attn_mean + uncertainty*r

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_mean, uncertainty

class Attention_tanh(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num, act = "sigmoid"):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        # self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        # elif act == "logexp":

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

    def norm_uncertainty(self, uncertainty):
        # uncertainty = torch.log(torch.exp(uncertainty))
        uncertainty = (torch.tanh(uncertainty) + 1)/2
        # uncertainty = self.act(uncertainty)
        return uncertainty

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, use_gate = False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        
        qk = (q @ k.transpose(-2, -1)) 

        attn = qk * self.scale
        attn_mean = attn.softmax(dim=-1)

        # uncertainty = self.data_uncertainty(qk)   # -inf, inf
        uncertainty = self.norm_uncertainty(qk)  # 0, 1

        if True:
            a, b, c, d = uncertainty.shape
            r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = (r>uncertainty)
            attn = attn_mean*mask
        else:
            # attn = attn*uncertainty
            r = torch.randn_like(uncertainty).to(uncertainty)
            attn = attn_mean + uncertainty*r

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_mean, uncertainty

class Attention_drop_out(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num, act = "sigmoid"):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(0.5)
        self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num


        if act == "sigmoid":
            self.act = nn.Sigmoid()
        # elif act == "logexp":

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

        print("Warning: This is directly dropout block")

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, use_gate = False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        
        qk = (q @ k.transpose(-2, -1)) 

        if self.repeat_num is not None:
            pass
        else:
            attn = qk * self.scale
            attn_mean = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn_mean, None

class Attention_Stage_2(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

    def norm_uncertainty(self, uncertainty):
        # uncertainty = torch.log(torch.exp(uncertainty))
        uncertainty = (torch.tanh(uncertainty) + 1)/2
        return uncertainty

    @property
    def unwrapped(self):
        return self

    def act_gradient(self, G = True):
        self.proj.weight.requires_grad = G
        self.proj.bias.requires_grad = G

    def forward(self, x, mask=None, stage = 1):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        # simplely use uncertainty
        if stage == 0:
            self.act_gradient(True)
            qk = (q @ k.transpose(-2, -1)) 
            uncertainty = self.data_uncertainty(qk)
            uncertainty = self.norm_uncertainty(uncertainty)
            a, b, c, d = uncertainty.shape
            r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = (r>uncertainty)

            attn = qk * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn*mask

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn
        # not use uncertainty
        elif stage == 1:
            self.act_gradient(True)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn
        # only train uncertainty
        elif stage == 2:
            self.act_gradient(False)
            q, k, v = q.detach(), k.detach(), v.detach()
            # x = x.detach()

            qk = (q @ k.transpose(-2, -1)) 
            uncertainty = self.data_uncertainty(qk)
            uncertainty = self.norm_uncertainty(uncertainty)
            a, b, c, d = uncertainty.shape
            r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = (r>uncertainty)

            attn = qk * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn*mask

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x, attn

class Attention_gumbel(nn.Module):
    def __init__(self, dim, heads, dropout, repeat_num, act = "sigmoid"):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.data_uncertainty = nn.Conv2d(heads, heads, kernel_size = 1, stride=1)
        self.repeat_num = repeat_num

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        # elif act == "logexp":

        if repeat_num is not None:
            print("UNCERTAINTY: The uncertainty block will process for ", repeat_num, " times")

    def norm_uncertainty(self, uncertainty):
        uncertainty = torch.log(torch.exp(uncertainty))
        uncertainty = (torch.tanh(uncertainty) + 1)/2
        # uncertainty = self.act(uncertainty)
        return uncertainty

    def get_gumbel_mask(self, x, hard):
        # get tensor [B, C, Hh, Ww, 2]
        y = 1-x  
        x = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        # gumbel softmax on last dim
        z = F.gumbel_softmax(x, dim=-1, hard=hard)
        z = z[:, :, :, :, 1]
        return z

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, use_gate = False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        
        qk = (q @ k.transpose(-2, -1)) 

        attn = qk * self.scale
        attn_mean = attn.softmax(dim=-1)

        uncertainty = self.data_uncertainty(qk)   # -inf, inf
        uncertainty = self.norm_uncertainty(uncertainty)  # 0, 1

        if True:
            # a, b, c, d = uncertainty.shape
            # r = torch.rand([a, b, c, d]).to(uncertainty)
            mask = self.get_gumbel_mask(uncertainty, hard=True)
            attn = attn_mean*mask
        else:
            # attn = attn*uncertainty
            r = torch.randn_like(uncertainty).to(uncertainty)
            attn = attn_mean + uncertainty*r

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_mean, uncertainty

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, with_ut = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, with_ut = with_ut)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, use_gate = None):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block_data(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, block_type, repeat_num = None):
        super().__init__()
        self.repeat_num = repeat_num
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if block_type == "block_data":
            self.attn = Attention_data(dim, heads, dropout, repeat_num)
        elif block_type == "block_dropout":
            self.attn = Attention_drop_out(dim, heads, dropout, repeat_num)
        elif block_type == "block_relu":
            self.attn = Attention_relu(dim, heads, dropout, repeat_num) 
        elif block_type == "block_tanh":
            self.attn = Attention_tanh(dim, heads, dropout, repeat_num) 
        elif block_type == "block_gumbel":
            self.attn = Attention_gumbel(dim, heads, dropout, repeat_num)    
        else:
            raise Exception("The attention block not implement")
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, use_gate = True):
        if self.repeat_num is not None:
            y_list, attn_list = self.attn(self.norm1(x), mask)
            if return_attention:
                return attn_list
            x_list = []
            for y in y_list:
                xx = x + self.drop_path(y)
                xx = xx + self.drop_path(self.mlp(self.norm2(xx)))
                x_list.append(x)
            return x_list

        else:
            y, attn, ut = self.attn(self.norm1(x), mask, use_gate = use_gate)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            if return_attention:
                return x, attn, ut
            return x

class Block_stage_2(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, repeat_num = None):
        super().__init__()
        self.repeat_num = repeat_num
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention_data(dim, heads, dropout, repeat_num)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, stage = 1):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
