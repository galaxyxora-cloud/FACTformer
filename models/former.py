
from models.layer import PatchEmbed, PatchUnEmbed
from models.transformer_block import *
from models.DegenerativePerceptionMoudel import DPM


class _EncoderSkipUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.dw_conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.PReLU()

        self.pw_conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.pool(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        return x
class DahazeFormer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 out_chans=3,
                 embed_dims=[32, 64, 128, 128, 64, 32],
                 depths=[1, 2, 4, 4, 2, 1],
                 num_heads=[2, 4, 8, 8, 4, 2],
                 num_points=[2, 4, 6, 6, 4, 2],
                 attn_drop=0.00,
                 proj_drop=0.01,
                 norm_layer=nn.BatchNorm2d):

        super(DahazeFormer, self).__init__()

        self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.encoder_level1 = BasicLayer(
            dim=embed_dims[0],
            depth=depths[0],
            num_heads=num_heads[0],
            num_points=num_points[0],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.down1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])  # 32-->64
        self.encoder_level2 = BasicLayer(
            dim=embed_dims[1],
            depth=depths[1],
            num_heads=num_heads[1],
            num_points=num_points[1],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.down2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])  # 64-->128
        self.encoder_level3 = BasicLayer(
            dim=embed_dims[2],
            depth=depths[2],
            num_heads=num_heads[2],
            num_points=num_points[2],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )

        self.enc_skip1 = _EncoderSkipUnit(in_chans, embed_dims[1])
        self.enc_skip2 = _EncoderSkipUnit(embed_dims[1], embed_dims[2])
        self.enc_proj12 = nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=1, bias=False)
        self.enc_proj23 = nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=1, bias=False)
        self.decoder_level1 = BasicLayer(
            dim=embed_dims[3],
            depth=depths[3],
            num_heads=num_heads[3],
            num_points=num_points[3],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.fusion1 = DPM(in_channel=embed_dims[3])
        self.up1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.decoder_level2 = BasicLayer(
            dim=embed_dims[4],
            depth=depths[4],
            num_heads=num_heads[4],
            num_points=num_points[4],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.fusion2 = DPM(in_channel=embed_dims[4])
        self.up2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])
        self.decoder_level3 = BasicLayer(
            dim=embed_dims[5],
            depth=depths[5],
            num_heads=num_heads[5],
            num_points=num_points[5],
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.fusion3 = DPM(in_channel=embed_dims[5])
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(embed_dims[5], 3, 7, padding=0, bias=True),
        )


    def forward(self, x):

        skip0 = x

        x = self.patch_embed(x)
        x = self.encoder_level1(x)
        skip1 = x

        sk1 = self.enc_skip1(skip0)
        x = self.down1(x)
        x = self.enc_proj12(torch.cat([x, sk1], dim=1))

        x = self.encoder_level2(x)
        skip2 = x

        sk2 = self.enc_skip2(sk1)
        x = self.down2(x)
        x = self.enc_proj23(torch.cat([x, sk2], dim=1))
        x = self.encoder_level3(x)
        skip3 = x

        x = self.decoder_level1(x)
        fusion1 = self.fusion1(skip3, x)
        x = self.up1(fusion1)
        x = self.decoder_level2(x)
        fusion2 = self.fusion2(skip2, x)
        x = self.up2(fusion2)
        x = self.decoder_level3(x)
        fusion3 = self.fusion3(skip1, x)
        out_main = self.final(fusion3)
        return out_main + skip0

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def build_net():
    return DahazeFormer()

