import torch
from collections import OrderedDict

# Load the pre-trained DINOv2 model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()

# Define the indices of the transformer blocks you want to extract outputs from
return_block_indices = [3, 6, 9, 11]  # Example indices

# Prepare a dummy input tensor
x = torch.randn(1, 3, 224, 224)  # Adjust the size as needed

# Pass the input through the patch embedding and positional encoding
x = model.patch_embed(x)
cls_token = model.cls_token.expand(x.size(0), -1, -1)
x = torch.cat((cls_token, x), dim=1)
x = x + model.pos_embed[:, :x.size(1), :]
x = model.pos_drop(x)

# Initialize an ordered dictionary to store outputs
outputs = OrderedDict()

# Iterate through the transformer blocks
for i, blk in enumerate(model.blocks):
    x = blk(x)
    if i in return_block_indices:
        outputs[f'block_{i}'] = x

# Apply the final normalization
x = model.norm(x)
outputs['final'] = x

# Now, 'outputs' contains the outputs from the specified blocks and the final output
