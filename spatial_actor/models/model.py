from einops import rearrange
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork

from spatial_actor.models.modules.attn import (
    Conv2DBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
)
from spatial_actor.models.modules.backbone import (
    load_clip,
    load_imagenet_res50,
)
from spatial_actor.models.modules.convex_up import ConvexUpSample


class GateFuser(nn.Module):
    """
    Lightweight two-branch feature fusion module that dynamically adjusts the fusion ratio through gating.
    """
    def __init__(self, in_ch_real, in_ch_da, mid_ch=128):
        """
        Inputs:
            in_ch_real: Integer, channel dimension of real Depth features.
            in_ch_da: Integer, channel dimension of Depth Anything features.
            mid_ch: Integer, channel dimension of the intermediate layer.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch_real + in_ch_da, mid_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, feat_r, feat_d):
        """
        Inputs:
            feat_r: Tensor, real Depth features of shape (B, C_real, H, W).
            feat_d: Tensor, Depth Anything features of shape (B, C_da, H, W), resized to match feat_r.
        Returns:
            fused: Tensor, fused features of shape (B, C_real, H, W).
        """
        x_cat = torch.cat([feat_r, feat_d], dim=1)
        x_mid = F.relu(self.bn1(self.conv1(x_cat)), inplace=True)
        gating_map = torch.sigmoid(self.conv2(x_mid))  # gating map
        alpha_r = gating_map  # (B, 1, H, W)
        alpha_d = 1 - alpha_r  # (B, 1, H, W)

        fused = alpha_r * feat_r + alpha_d * feat_d
        return fused


class FixedPositionalEncoding(nn.Module):
    def __init__(self, feat_per_dim: int, feat_scale_factor: int):
        super().__init__()
        self.feat_scale_factor = feat_scale_factor
        # shape [1, feat_per_dim // 2]
        div_term = torch.exp(
            torch.arange(0, feat_per_dim, 2) * (-math.log(10000.0) /
                                                feat_per_dim)
        ).unsqueeze(0)
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        :param x: Tensor, shape [batch_size, input_dim]
        :return: Tensor, shape [batch_size, input_dim * feat_per_dim]
        """
        assert len(x.shape) == 2
        batch_size, input_dim = x.shape
        x = x.view(-1, 1)
        x = torch.cat((
            torch.sin(self.feat_scale_factor * x * self.div_term),
            torch.cos(self.feat_scale_factor * x * self.div_term)), dim=1)
        x = x.view(batch_size, -1)
        return x


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class SemanticGuidedGeometricModule(nn.Module):
    def __init__(
            self,
            dep_exp_type,
            fpn_fuse_dim,
            align_loss
    ):
        super().__init__()

        self.fpn_fuse_dim = fpn_fuse_dim
        self.align_loss = align_loss

        model_type = dep_exp_type.replace('DA-', '')
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        res_channels = [64, 256, 512, 1024, 2048]
        mid_channels = [64, 128, 256, 512, 512]

        self.depth_expert_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if self.align_loss > 0.0:
            self.depth_expert_align_proj = nn.Linear(
                model_configs[model_type]['out_channels'][-1],
                fpn_fuse_dim)

            self.geometric_align_fpn = FeaturePyramidNetwork(
                res_channels,
                fpn_fuse_dim)

        self.gate_proj_layers = nn.ModuleList([
            nn.Conv2d(
                model_configs[model_type]['out_channels'][-1],
                res_channels[i],
                kernel_size=1)
            for i in range(len(res_channels))
        ])

        self.gate_fuse_layers = nn.ModuleList([
            GateFuser(
                in_ch_real=res_channels[i],
                in_ch_da=res_channels[i],
                mid_ch=mid_channels[i])
            for i in range(len(res_channels))
        ])

        self.fpn_fuse = FeaturePyramidNetwork(
            [c * 2 for c in res_channels],
            fpn_fuse_dim
        )

    def forward(
            self,
            depth_expert,
            d0,
            semantic_feat,
            geometic_feat
    ):
        with torch.no_grad():
            rgb_de = self.depth_expert_norm(d0[:, 3:6])

            if depth_expert.n_blocks == 24:
                layer_ids_de = [11, 14, 17, 20, 23]
            elif depth_expert.n_blocks == 12:
                layer_ids_de = [3, 5, 7, 9, 11]
            else:
                raise NotImplementedError

            depth_expert_feat = depth_expert.get_intermediate_layers(
                x=rgb_de,
                n=layer_ids_de,
                reshape=True,
                return_class_token=False,
                norm=True
            )

        sgm_feat = {}
        for i, key in enumerate(['res1', 'res2', 'res3', 'res4', 'res5']):
            feat_r = geometic_feat[key]
            feat_e = depth_expert_feat[i]

            feat_e_proj = self.gate_proj_layers[i](feat_e)
            feat_d_resized = F.interpolate(feat_e_proj, size=(feat_r.shape[2], feat_r.shape[3]),
                                           mode='bilinear', align_corners=False)

            sgm_feat[key] = self.gate_fuse_layers[i](feat_r, feat_d_resized)

        spatial_feat = {}
        for k1, k2 in zip(semantic_feat.keys(), sgm_feat.keys()):
            spatial_feat[k1] = torch.cat([semantic_feat[k1], sgm_feat[k2]], dim=1)

        spatial_feat = self.fpn_fuse(spatial_feat)['res3']

        if self.align_loss > 0.0 and self.training:
            geometic_feat = self.geometric_align_fpn(geometic_feat)['res3']
            depth_expert_feat = F.interpolate(
                depth_expert_feat[-1],
                size=geometic_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)

            _b, _c, _h, _w = depth_expert_feat.shape
            depth_expert_feat = depth_expert_feat.reshape(_b, _c, _h * _w).permute(0, 2, 1)
            depth_expert_feat = self.depth_expert_align_proj(depth_expert_feat)
            depth_expert_feat = depth_expert_feat.permute(0, 2, 1).reshape(_b, self.fpn_fuse_dim, _h, _w)

        return spatial_feat, geometic_feat, depth_expert_feat


class SpatialTransformer(nn.Module):
    def __init__(
            self,
            spt_view_layers,
            spt_scene_layers,
            attn_dim,
            attn_heads,
            attn_dim_head,
            fpn_fuse_dim,
            im_channels,
            input_dim_before_seq,
            lang_len,
            activation,
            weight_tie_layers,
            attn_dropout,
            xops,
    ):
        super().__init__()

        self.lang_len = lang_len

        self.spatial_pe_layer = RotaryPositionEncoding3D(fpn_fuse_dim)

        self.spatial_feat_proj = Conv2DBlock(
            fpn_fuse_dim,
            im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.fc_bef_attn = DenseBlock(
            input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )

        self.fc_aft_attn = DenseBlock(
            attn_dim,
            input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=xops,
            ),
        )

        get_ffn = lambda: PreNorm(
            attn_dim,
            FeedForward(attn_dim)
        )

        get_attn, get_ffn = map(cache_fn, (get_attn, get_ffn))
        cache_args = {"_cache": weight_tie_layers}

        self.spt_view_interact = nn.ModuleList([])
        for _ in range(spt_view_layers):
            self.spt_view_interact.append(
                nn.ModuleList([get_attn(**cache_args), get_ffn(**cache_args)])
            )

        self.spt_scene_interact = nn.ModuleList([])
        for _ in range(spt_scene_layers):
            self.spt_scene_interact.append(
                nn.ModuleList([get_attn(**cache_args), get_ffn(**cache_args)])
            )

    def forward(
            self,
            d0,
            num_img,
            spatial_feat,
            lang_feat,
            proprio_feat,
    ):
        _b, _c, _h, _w = spatial_feat.shape
        bs = _b // num_img

        # spatial position encoding
        xyz = d0[:, 0:3]
        xyz = F.interpolate(xyz, size=(_h, _w), mode='bilinear', align_corners=False)
        xyz = xyz.view(bs, num_img, 3, _h, _w).permute(0, 1, 3, 4, 2).reshape(bs, num_img*_h*_w, 3)
        pe = self.spatial_pe_layer(xyz)
        pe_cos, pe_sin = pe[..., 0], pe[..., 1]

        _b, _c, _h, _w = spatial_feat.shape

        spatial_feat = spatial_feat.view(bs, num_img, _c, _h, _w).permute(0, 1, 3, 4, 2)
        spatial_feat = spatial_feat.reshape(bs, num_img*_h*_w, _c)
        spatial_feat = RotaryPositionEncoding.embed_rotary(spatial_feat, pe_cos, pe_sin)

        spatial_feat = spatial_feat.view(bs*num_img, _h, _w, _c).permute(0, 3, 1, 2)
        spatial_feat = self.spatial_feat_proj(spatial_feat)

        _, _c, _h, _w = spatial_feat.shape
        spatial_feat = spatial_feat.view(bs, num_img, _c, _h, _w).permute(0, 2, 1, 3, 4)

        if proprio_feat is not None:
            spatial_feat = torch.cat([spatial_feat, proprio_feat], dim=1)

        # channel last
        spatial_feat = rearrange(spatial_feat, "b d ... -> b ... d")

        # save original shape of input
        spatial_feat_orig_shape = spatial_feat.shape

        spatial_feat = rearrange(spatial_feat, "b ... d -> b (...) d")

        lang_feat = lang_feat.view(bs, self.lang_len, -1)
        num_lang_tok = lang_feat.shape[1]
        spatial_feat = torch.cat((lang_feat, spatial_feat), dim=1)

        x = self.fc_bef_attn(spatial_feat)

        lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

        # view-level interaction
        imgx = imgx.reshape(bs * num_img, _h * _w, -1)

        for attn, ffn in self.spt_view_interact:
            imgx = attn(imgx) + imgx
            imgx = ffn(imgx) + imgx

        # scene-level interaction
        imgx = imgx.view(bs, num_img * _h * _w, -1)
        x = torch.cat((lx, imgx), dim=1)

        for attn, ffn in self.spt_scene_interact:
            x = attn(x) + x
            x = ffn(x) + x

        # throwing away the language embeddings
        x = x[:, num_lang_tok:]

        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *spatial_feat_orig_shape[1:-1], x.shape[-1])
        x = rearrange(x, "b ... d -> b d ...")

        return x


class SpatialActor(nn.Module):
    def __init__(
        self,
        sem_enc_type,
        geo_enc_type,
        dep_exp_type,
        lang_dim,
        lang_len,
        add_proprio,
        proprio_dim,
        proprio_cat_dim,
        spt_view_layers,
        spt_scene_layers,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        img_patch_size,
        final_dim,
        img_feat_dim,
        num_rot,
        feat_dim,
        img_size,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        xops,
        renderer,
        no_feat,
        align_loss,
        **kwargs,
    ):
        """
        Single SpatialActor module.

        Parameters
        ----------
        sem_enc_type : str
            Type of semantic encoder.
        geo_enc_type: str
            Tye of geometric encoder.
        dep_exp_type : str
            Type of depth expert.
        lang_dim : int
            Dimensionality of language features.
        lang_len : int
            Maximum number of language tokens.
        add_proprio : bool
            Whether to include proprioceptive information.
        proprio_dim : int
            Dimensionality of raw proprioceptive input.
        proprio_cat_dim : int
            Dimensionality after concatenating proprioceptive features.
        spt_view_layers : int
            Number of spatial Transformer layers for view-level interactions.
        spt_scene_layers : int
            Number of spatial Transformer layers for scene-level interactions.
        im_channels : int
            Number of intermediate feature channels.
        attn_dim : int
        attn_heads : int
        attn_dim_head : int
        activation : str
        weight_tie_layers : bool
        attn_dropout : float
        img_patch_size : int
            Initial patch size for image tokenization.
        final_dim : int
            Dimensionality of the final output features.
        img_feat_dim : int
            Output feature dimensionality of the semantic encoder.
        num_rot : int
            Number of discrete rotation bins for rotation prediction.
        feat_dim : int
            General feature dimensionality used within the network.
        img_size : int
            Spatial resolution (image side length in pixels) used for rendering.
        add_corr : bool
            Whether to include correspondence features (e.g., cross-view or 2D–3D correspondences).
        norm_corr : bool
            Whether to normalize correspondence values.
            This is important when coordinates are outside the range [-1, 1],
            such as in two-stage SpacialActor settings.
        add_pixel_loc : bool
            Whether to explicitly add pixel location encodings.
        add_depth : bool
            Whether to include depth as an additional input feature.
        xops : int
            Strategy for XYZ prediction:
                - 0: independent discrete prediction for x, y, z
                - 1: conditional prediction where x, y, z depend on each other
        renderer : nn.Module
            Renderer module for point cloud or feature rendering.
        no_feat : bool
            If True, only action predictions are returned without intermediate features.
        align_loss : float
            Weight of the alignment loss.
        """

        super().__init__()

        self.sem_enc_type = sem_enc_type
        self.geo_enc_type = geo_enc_type
        self.dep_exp_type = dep_exp_type

        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.proprio_cat_dim = proprio_cat_dim

        self.spt_view_layers = spt_view_layers
        self.spt_scene_layers = spt_scene_layers
        self.im_channels = im_channels

        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.img_feat_dim = img_feat_dim
        self.num_rot = num_rot
        self.img_size = img_size

        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth

        self.no_feat = no_feat
        self.align_loss = align_loss

        self.renderer = renderer

        self.num_img = self.renderer.num_img

        if self.add_proprio:
            self.input_dim_before_seq = self.im_channels + self.proprio_cat_dim
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing
        self.input_preprocess = lambda x: x

        # semantic encoder
        if self.sem_enc_type in ['CLIP-RN50', 'CLIP-RN101']:
            self.semantic_encoder, self.rgb_norm = load_clip(type=self.sem_enc_type.replace('CLIP-', ''))
        elif self.sem_enc_type in ['RN50']:
            self.semantic_encoder, self.rgb_norm = load_imagenet_res50(pretrained=True)
        else:
            raise NotImplementedError

        self.add_module('semantic_encoder', self.semantic_encoder)

        # geometric encoder
        if self.geo_enc_type in ['RN50']:
            self.geometric_encoder, self.depth_norm = load_imagenet_res50(pretrained=True)
        else:
            raise NotImplementedError

        self.add_module('geometric_encoder', self.geometric_encoder)

        self.fpn_fuse_dim = self.im_channels - self.im_channels % 6

        # lang projector
        self.lang_proj = DenseBlock(
            self.lang_dim,
            self.input_dim_before_seq,
            norm="group",
            activation=activation,
        )

        # proprio projector
        if self.add_proprio:
            self.proprio_proj = DenseBlock(
                self.proprio_dim,
                self.proprio_cat_dim,
                norm="group",
                activation=activation,
            )

        self.sem_guide_geo_module = SemanticGuidedGeometricModule(
            dep_exp_type=self.dep_exp_type,
            fpn_fuse_dim=self.fpn_fuse_dim,
            align_loss=self.align_loss,
        )

        self.spatial_transformer = SpatialTransformer(
            spt_view_layers=spt_view_layers,
            spt_scene_layers=spt_scene_layers,
            attn_dim=attn_dim,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            fpn_fuse_dim=self.fpn_fuse_dim,
            im_channels=im_channels,
            input_dim_before_seq=self.input_dim_before_seq,
            lang_len=lang_len,
            activation=activation,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            xops=xops,
        )

        self.trans_head = ConvexUpSample(
            in_dim=self.input_dim_before_seq,
            out_dim=1,
            up_ratio=self.img_patch_size,
        )

        if not self.no_feat:
            fc_head_dim = 0
            fc_head_dim += self.input_dim_before_seq
            fc_head_dim += self.input_dim_before_seq

            def get_fc_head(
                _feat_in_size,
                _feat_out_size,
                _fc_head_dim=fc_head_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _fc_head_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _fc_head_dim),
                    nn.ReLU(),
                    nn.Linear(_fc_head_dim, _fc_head_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_fc_head_dim // 2, _feat_out_size),
                ]
                fc_head = nn.Sequential(*layers)
                return fc_head

            feat_out_size = feat_dim

            assert self.num_rot * 3 <= feat_out_size

            feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
            if feat_out_size_ex_rot > 0:
                self.fc_head_ex_rot = get_fc_head(
                    self.num_img * fc_head_dim, feat_out_size_ex_rot
                )

            self.fc_head_init_bn = nn.BatchNorm1d(self.num_img * fc_head_dim)
            self.fc_head_pe = FixedPositionalEncoding(
                self.num_img * fc_head_dim, feat_scale_factor=1
            )
            self.fc_head_x = get_fc_head(self.num_img * fc_head_dim, self.num_rot)
            self.fc_head_y = get_fc_head(self.num_img * fc_head_dim, self.num_rot)
            self.fc_head_z = get_fc_head(self.num_img * fc_head_dim, self.num_rot)

        from point_renderer.rvt_ops import select_feat_from_hm
        global select_feat_from_hm

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        depth_expert=None,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param wpt_local: (bs, 3)
        :param rot_x_y: (bs, 2)
        :param cams: camera information
        :param depth_expert: depth expert model
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        assert num_img == self.num_img
        assert h == w == self.img_size

        # preprocess
        # (bs * num_img, im_channels, h, w)
        img = img.view(bs * num_img, img_feat_dim, h, w)
        d0 = self.input_preprocess(img)

        # semantic encoder
        rgb = d0[:, 3:6]
        rgb = self.rgb_norm(rgb)

        semantic_feat = self.semantic_encoder(rgb)

        # geometric encoder
        depth = d0[:, 6:7].clone()
        depth = torch.cat([depth, depth, depth], dim=1)

        depth_min = depth.amin(dim=[2, 3], keepdim=True)
        depth_max = depth.amax(dim=[2, 3], keepdim=True)
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth_normalized = self.depth_norm(depth_normalized)

        geometic_feat = self.geometric_encoder(depth_normalized)

        # semantic guided geometric module
        spatial_feat, geometic_feat, depth_expert_feat = self.sem_guide_geo_module(
            depth_expert=depth_expert,
            d0=d0,
            semantic_feat=semantic_feat,
            geometic_feat=geometic_feat,
        )

        # language projector
        lang_feat = self.lang_proj(
            lang_emb.view(bs * self.lang_len, self.lang_dim)
        )

        _b, _c, _h, _w = spatial_feat.shape

        # proprio projector
        if self.add_proprio:
            proprio_feat = self.proprio_proj(proprio)
            proprio_feat = proprio_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_img, _h, _w)

        # spatial transformer
        x = self.spatial_transformer(
            d0=d0,
            num_img=num_img,
            spatial_feat=spatial_feat,
            lang_feat=lang_feat,
            proprio_feat=proprio_feat if self.add_proprio else None,
        )

        # action head
        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat) # (b, c * num_img)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, _h, _w
            )
        ) # (b * num_img, c, _h, _w)

        trans = self.trans_head(x) # (b * num_img, 1, h, w)
        trans = trans.view(bs, self.num_img, h, w)

        # get wpt_local while testing
        if not self.training:
            wpt_local = self.get_wpt(
                out={"trans": trans.clone().detach()},
                dyn_cam_info=None,
            )

        # projection
        # (bs, 1, num_img, 2)
        wpt_img = self.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            dyn_cam_info=None,
        )
        wpt_img = wpt_img.reshape(bs * self.num_img, 2)

        wpt_img = (wpt_img / self.img_patch_size).unsqueeze(1)

        if not self.no_feat:
            assert (
                0 <= wpt_img.min() and wpt_img.max() <= x.shape[-1]
            ), print(wpt_img, x.shape)

            _feat = select_feat_from_hm(wpt_img, x)[0].view(bs, -1)
            feat.append(_feat) # heatmap max point，wpt
            feat = torch.cat(feat, dim=-1)

            feat = feat.unsqueeze(1)

            # features except rotation
            feat_ex_rot = self.fc_head_ex_rot(feat) # (bs, 2*num_view*c) -> (bs, 4)

            # batch normalized features for rotation
            feat_rot = self.fc_head_init_bn(feat.permute(0, 2, 1)).permute(0, 2, 1)
            feat_x = self.fc_head_x(feat_rot) # (bs, 2*num_view*c) -> (bs, 72)

            if self.training:
                rot_x = rot_x_y[..., 0].view(bs, 1) # 0-71, represent which bin
            else:
                # sample with argmax
                rot_x = feat_x[:,0].argmax(dim=1, keepdim=True)

            rot_x_pe = self.fc_head_pe(rot_x).unsqueeze(1)
            feat_y = self.fc_head_y(feat_rot + rot_x_pe) # (bs, 2*num_view*c) -> (bs, 72)

            if self.training:
                rot_y = rot_x_y[..., 1].view(bs, 1)
            else:
                rot_y = feat_y[:,0].argmax(dim=1, keepdim=True)
            rot_y_pe = self.fc_head_pe(rot_y).unsqueeze(1)
            feat_z = self.fc_head_z(feat_rot + rot_x_pe + rot_y_pe) # (bs, 2*num_view*c) -> (bs, 72)
            out = {
                "feat_ex_rot": feat_ex_rot,
                "feat_x": feat_x,
                "feat_y": feat_y,
                "feat_z": feat_z,
            }
        else:
            out = {}

        out.update({"trans": trans})

        if self.align_loss > 0.0 and self.training:
            align_feats = {}
            align_feats.update({'geometic_feat': geometic_feat})
            align_feats.update({'depth_expert_feat': depth_expert_feat})
            align_feats.update({'align_loss': self.align_loss})

            out.update({'align_feats': align_feats})

        return out


    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output
        :param out: output
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if pred_wpt.shape[1] > 1:
            pred_wpt = torch.mean(pred_wpt, 1)
        else:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
