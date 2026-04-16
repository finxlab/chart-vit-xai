import os
import h5py
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILImage
from torchvision.models import vit_b_32, vit_b_16

# ─────────────────────────────────────────────
# 0. Model builder
# ─────────────────────────────────────────────
def build_vit(patch_size=32, num_encoder_layers=2, num_classes=2):
    if patch_size == 32:
        model = vit_b_32(weights=None)
    elif patch_size == 16:
        model = vit_b_16(weights=None)
    model.encoder.layers = nn.Sequential(
        *[model.encoder.layers[i] for i in range(num_encoder_layers)]
    )
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
    nn.init.zeros_(model.heads.head.bias)
    return model

# ─────────────────────────────────────────────
# 1. Load image
# ─────────────────────────────────────────────
def load_image(h5_path, num_idx):
    with h5py.File(h5_path, 'r') as f:
        image_np = f['images'][num_idx]
        date     = int(f['dates'][num_idx])
        permno   = int(f['permnos'][num_idx])
    image_display = np.transpose(image_np, (1, 2, 0))
    image_tensor  = torch.from_numpy(image_np).float() / 255.0
    return image_tensor, image_display, date, permno

# ─────────────────────────────────────────────
# 2. Manual forward
# ─────────────────────────────────────────────
def forward_and_get_attn(model, image_tensor):
    x = image_tensor.unsqueeze(0)
    x = model._process_input(x)
    cls_token = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = model.encoder.dropout(x + model.encoder.pos_embedding)

    attn_list = []
    for layer in model.encoder.layers:
        y = layer.ln_1(x)
        B, N, C = y.shape
        num_heads = layer.self_attention.num_heads
        head_dim  = C // num_heads
        scale     = head_dim ** -0.5

        qkv = torch.nn.functional.linear(
            y, layer.self_attention.in_proj_weight,
            layer.self_attention.in_proj_bias
        )
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn.retain_grad()
        attn_list.append(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = torch.nn.functional.linear(
            out, layer.self_attention.out_proj.weight,
            layer.self_attention.out_proj.bias
        )
        x = x + out
        x = x + layer.mlp(layer.ln_2(x))

    x      = model.encoder.ln(x)
    logits = model.heads(x[:, 0])
    return logits, attn_list

# ─────────────────────────────────────────────
# 3. Chefer relevancy
# ─────────────────────────────────────────────
def chefer_relevancy(model, image_tensor, class_idx=1, clip=95):
    model.eval()
    logits, attn_list = forward_and_get_attn(model, image_tensor)
    model.zero_grad()
    logits[0, class_idx].backward()

    num_tokens = attn_list[0].shape[-1]
    device     = image_tensor.device
    R = torch.eye(num_tokens, device=device)

    for attn in attn_list:
        grad = attn.grad
        if grad is None:
            continue
        cam = (grad * attn)[0]
        cam = torch.relu(cam)
        cam = cam.mean(dim=0)
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
        cam = cam + torch.eye(num_tokens, device=device)
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
        R   = cam @ R

    mask = R[0, 1:].detach().cpu().numpy().astype(np.float32)
    mask = np.clip(mask, 0, np.percentile(mask, clip))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    grid_size = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(grid_size, grid_size)
    mask = np.array(
        PILImage.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), PILImage.BILINEAR)
    ) / 255.0
    return mask

# ─────────────────────────────────────────────
# 4. Overlay
# ─────────────────────────────────────────────
def make_overlay(image_display, heatmap, alpha=0.55, cmap='jet'):
    c   = cm.get_cmap(cmap)(heatmap)[:, :, :3]
    img = image_display / 255.0
    out = alpha * c + (1 - alpha) * img
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

# ─────────────────────────────────────────────
# 5. Save clean image (no border, no text)
# ─────────────────────────────────────────────
def save_clean(img_array, out_path, dpi=400):
    h, w = img_array.shape[:2]
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_array)
    ax.axis('off')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path',    type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_idx',    type=int, required=True)
    parser.add_argument('--class_idx',  type=int, default=0,
                        help='1=Up, 0=Down')
    parser.add_argument('--clip',       type=int, default=95)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='chefer_selected')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load
    image_tensor, image_display, date, permno = load_image(args.h5_path, args.num_idx)
    image_tensor = image_tensor.to(device)
    print(f"PERMNO={permno} | date={date} | num_idx={args.num_idx}")

    # Load model
    model = build_vit(patch_size=args.patch_size, num_encoder_layers=args.num_layers)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        probs  = torch.softmax(logits.float(), dim=1)[0]
    print(f"Prob Up={probs[1].item():.4f} | Prob Down={probs[0].item():.4f}")

    # Chefer
    label = "up" if args.class_idx == 1 else "down"
    print(f"Computing relevancy ({label})...")
    rel_map = chefer_relevancy(model, image_tensor,
                               class_idx=args.class_idx, clip=args.clip)
    overlay_img = make_overlay(image_display, rel_map)

    orig_path   = os.path.join(args.output_dir, f'orig_{permno}_{date}.png')
    chefer_path = os.path.join(args.output_dir, f'chefer_{label}_{permno}_{date}.png')

    save_clean(image_display, orig_path)
    save_clean(overlay_img,   chefer_path)

    print(f"Saved → {orig_path}")
    print(f"Saved → {chefer_path}")


if __name__ == '__main__':
    import sys
    sys.argv = [
        'chefer.py',
        '--h5_path',    'DB/test/rgb_30d_test.h5',
        '--model_path', 'experiments/ViT_B32_30d/enc2_batch1024_rs2ratio0.1_lr0.0001_wd0.05/seed92/best_model.pth',
        '--num_idx',    '488310',
        '--class_idx',  '1',       # 0=Down, 1=Up
        '--output_dir', 'chefer_selected',
    ]
    main()