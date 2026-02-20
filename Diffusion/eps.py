# 前向过程: q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I) 逐步增加高斯分布的噪声,是一个马尔科夫链的过程
# 可简化成从初始状态到任意时间步的 x_t = √(ᾱ_t) x₀ + √(1-ᾱ_t) ε
# β:预定义的噪声比例参数,在传播过程中逐渐增加
# 前向扩散核: FDK 是一个高斯分布
# 反向扩散核: RDK 被证明是FDK的一个泛函,参数由神经网络训练得到
# 用来训练DDPM的最终损失函数,它只是前向过程中加入的噪声与模型预测噪声之间的“均方误差”

import torch
import torch.nn as nn


#实现一个简单的前向加噪过程
class SimpleDiffusion(nn.Module):
    def __init__(
            self,
            time_steps: int = 1000,
            image_shape: tuple = (3, 64, 64), # channel , picture_size
            device: str = 'cpu'
    ):
        super().__init__()
        self.time_steps = time_steps
        self.image_shape = image_shape
        self.device = device
        self.initialize()

    def initialize(self):
        self.beta = self.get_betas() # β
        self.alpha = 1 - self.beta # α

        self.sqrt_beta = torch.sqrt(self.beta) # √β
        self.alpha_cumulative = torch.cat([torch.ones(1, device=self.device), torch.cumprod(self.alpha, dim=0)])[:-1] # ∏α
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative) # √∏α
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha) # 1/√α_t
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative) # √(1-∏α)

        # 将所有张量移到指定设备
        for attr_name in ['beta', 'alpha', 'sqrt_beta', 'alpha_cumulative',
                          'sqrt_alpha_cumulative', 'one_by_sqrt_alpha',
                          'sqrt_one_minus_alpha_cumulative']:
            if hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self, attr_name).to(self.device))

    def get_betas(self):
        scale = 1000/self.time_steps
        beta_star = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(beta_star,beta_end,self.time_steps,dtype=torch.float32,device=self.device)

    def get(self,param: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(min=0, max=self.time_steps - 1)
        return param[t][:, None, None, None]

    def forward(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor
    ) -> tuple: #tuple -> [image_with_noise , noise]
        eps = torch.randn_like(x0) # Noise

        mean = self.get(self.sqrt_alpha_cumulative,t=timesteps) * x0
        std_dev = self.get(self.sqrt_one_minus_alpha_cumulative, t=timesteps)

        sample = mean + std_dev * eps

        return sample,eps

    def to(self, device: str):
        """将模型所有张量移到指定设备"""
        super().to(device)
        self.device = device

        # 将所有张量属性移到新设备
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.to(device))

        return self

def sample_timesteps(batch_size, time_steps, device) -> torch.Tensor:
    """随机采样时间步用于训练"""
    return torch.randint(0, time_steps, (batch_size,), device=device)


def load_image_pil_torch(image_path):
    from PIL import Image
    import torch
    from torchvision import transforms
    # 打开图片
    image = Image.open(image_path)

    # 定义转换管道
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整大小
        #transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量 [0,1]范围
        transforms.Normalize(  # 标准化（常用ImageNet统计量）
            mean=[0.485, 0.456, 0.406],  # RGB均值
            std=[0.229, 0.224, 0.225]  # RGB标准差
        )
    ])

    # 应用转换
    tensor = transform(image)  # 形状: [C, H, W]

    # 添加批次维度（如果需要）
    tensor = tensor.unsqueeze(0)  # 形状: [1, C, H, W]

    return tensor

def draw(x,timestep=None):
    import matplotlib.pyplot as plt

    batch_size,_,_,_ = x.shape

    if batch_size == 1:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, batch_size, figsize=(12, 4))

    for batch in range(batch_size):
        img = x[batch].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[batch].imshow(img)
        if timestep is not None:
            axes[batch].set_title(f"t={timestep[batch]}")
        axes[batch].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x0 = load_image_pil_torch('mount.png')

    _,channel,h,w = x0.shape
    batch_size = 6
    time_steps=999
    device = 'cpu'


    diffusion = SimpleDiffusion(
        time_steps=1000,
        image_shape=(channel, h, w),
        device=device
    )

    timestep = sample_timesteps(batch_size=batch_size,time_steps=time_steps,device=device)

    sample,eps = diffusion(x0 = x0,timesteps = timestep)

    draw(sample, timestep)







