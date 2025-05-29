import torch
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent / "deps" / "dinov2"
sys.path.append(str(repo_root))

from dinov2.models.vision_transformer import vit_large, vit_base       # source code class

# build the bare architecture (same kwargs they used for pre‑training)
model = vit_base(
    patch_size=14,
    img_size=518,        # 518×518 was used for ViT‑L/14
    init_values=1.0,
    block_chunks=0,      # 0 = no gradient checkpointing,
    num_register_tokens=4,
)

# load the backbone weights
state = torch.load("/home/jordan/omscs/ML4DED/models/dinov2_vitb14_reg4_pretrain.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()            # freeze for inference
