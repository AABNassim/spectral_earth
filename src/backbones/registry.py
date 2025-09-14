from .spec_resnet import SpecResNet50, ModifiedResNet50
from .vit import ViTSmall, ViTTiny
from .spec_vit import (
    SpecViTTiny,
    SpecViTSmall,
    SpecViTBase,
    SpecViTLarge,
    SpecViTHuge,
    SpecViTGiant,
)


BACKBONE_REGISTRY = {
    "spec_resnet50": SpecResNet50,
    "spec_vit_tiny": SpecViTTiny,
    "spec_vit_small": SpecViTSmall, 
    "spec_vit_base": SpecViTBase,
    "spec_vit_large": SpecViTLarge,
    "spec_vit_huge": SpecViTHuge,
    "spec_vit_giant": SpecViTGiant,
    "resnet50": ModifiedResNet50, 
    "vit_small": ViTSmall,
    "vit_tiny": ViTTiny,
}
