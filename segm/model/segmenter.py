import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        repeat_num = None,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.repeat_num = repeat_num

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im, use_gate = True):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, attn_mean, uncertainty = self.encoder(im, return_features=True, use_gate = use_gate)
        # print(type(x))
        

        if self.repeat_num is not None:
            # 如果返回的是list
            x_list = x

            # for i, x in enumerate(x_list):
            #     print(i, x.shape)

            for i, x in enumerate(x_list):
                num_extra_tokens = 1 + self.encoder.distilled
                x = x[:, num_extra_tokens:]
                

                masks = self.decoder(x, (H, W))

                masks = F.interpolate(masks, size=(H, W), mode="bilinear")
                masks = unpadding(masks, (H_ori, W_ori))
                if i == 0:
                    m_results = masks/self.repeat_num
                else:
                    m_results = m_results + masks/self.repeat_num
            
            return m_results
        else:

            # remove CLS/DIST tokens for decoding
            # print(x.shape)      # 4, 1025, 192
            num_extra_tokens = 1 + self.encoder.distilled
            x = x[:, num_extra_tokens:]
            

            masks = self.decoder(x, (H, W))

            masks = F.interpolate(masks, size=(H, W), mode="bilinear")
            masks = unpadding(masks, (H_ori, W_ori))

            return masks, attn_mean, uncertainty

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
