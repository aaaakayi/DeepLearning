import torch
import matplotlib.pyplot as plt
from eps import SimpleDiffusion
from U_Net import TimeEmbedding,U_net

def generate_samples_fixed(unet, time_embedding, diffusion, num_samples=16, device='cpu'):
    unet.eval()
    betas = diffusion.beta
    alphas = 1. - betas
    T = diffusion.time_steps

    # 累积乘积与训练保持一致: alpha_bar_t = prod_{i=0..t} alpha_i
    alphas_cumprod = diffusion.alpha_cumulative

    x = torch.randn(num_samples, 1, 32, 32, device=device)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = time_embedding(t_tensor)

            pred_noise = unet(x, t_emb)

            alpha_t = alphas[t].to(device)
            alpha_bar_t = alphas_cumprod[t].to(device)
            beta_t = betas[t].to(device)
            if t > 0:
                alpha_bar_prev = alphas_cumprod[t - 1].to(device)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # 分母加小常数防止除零
            coeff = (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t) + 1e-8)

            if t > 0:
                z = torch.randn_like(x)
                beta_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                sigma = torch.sqrt(beta_tilde)
            else:
                z = 0
                sigma = 0

            x = (1 / torch.sqrt(alpha_t)) * (x - coeff * pred_noise) + sigma * z

    images = (x + 1) / 2
    images = torch.clamp(images, 0, 1)
    return images

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet = U_net(num_layers=5, in_ch=1, out_ch=1).to(device)
    checkpoint = torch.load("checkpoint/diffusion_mnist.pth", map_location=device)
    if 'unet_ema_state_dict' in checkpoint:
        unet.load_state_dict(checkpoint['unet_ema_state_dict'], strict=False)
    else:
        unet.load_state_dict(checkpoint['unet_state_dict'])

    diffusion = SimpleDiffusion(
        time_steps=1000,
        image_shape=(1, 32, 32),
        device=device
    )

    time_embedding = TimeEmbedding().to(device)

    generated = generate_samples_fixed(
        unet = unet,
        time_embedding = time_embedding,
        diffusion = diffusion,
        num_samples = 16,
        device = device
    )

    # 显示图像
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.show()
