from timm.models.vision_transformer import VisionTransformer    



def ViTSmall(
    patch_size=128, token_patch_size=8, in_chans=202, num_classes=0,
    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, qkv_bias=True,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    **kwargs,  
):

    return VisionTransformer(
        img_size=patch_size, patch_size=token_patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        **kwargs
    )

def ViTTiny(
    patch_size=128, token_patch_size=8, in_chans=202, num_classes=0,
    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, qkv_bias=True,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    **kwargs,  
):

    return VisionTransformer(
        img_size=patch_size, patch_size=token_patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        **kwargs
    )
