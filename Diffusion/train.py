import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data as Data
import os

# 假设你的 SimpleDiffusion, U_net, TimeEmbedding 已经定义在单独的模块中
# 如果还没有，请将之前实现的类复制到此处或确保正确导入
from eps import SimpleDiffusion
from U_Net import U_net, TimeEmbedding


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    def apply_to(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            param.data.copy_(self.shadow[name])


def train_one_step(images, t, eps_model, time_embedding, unet, criterion, optimizer):
    """
    单步训练：接收一批图像和时间步，执行一次梯度更新
    """
    # 前向扩散：生成带噪图像和真实噪声
    noisy_images, noise = eps_model(images, t)

    # 时间嵌入
    t_emb = time_embedding(t)

    # U-Net 预测噪声
    unet.train()
    predicted_noise = unet(noisy_images, t_emb)

    # 计算损失
    loss = criterion(noise, predicted_noise)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    # -------------------- 配置参数 --------------------
    device = 'cpu'  # 使用 CPU
    timestep = 1000
    image_size = 32
    batch_size = 64
    num_epochs = 100
    learning_rate = 2e-4
    save_interval = 10  # 每10轮保存一次模型
    model_save_path = './checkpoint/diffusion_mnist.pth'
    ema_decay = 0.9999

    # -------------------- 数据准备 --------------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # 映射到 [-1,1]
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    trainLoader, validLoader = Data.random_split(
        dataset,
        lengths=[int(0.9 * len(dataset)),len(dataset) - int(0.9 * len(dataset))],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(
        trainLoader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # CPU 训练时 num_workers 设为0更稳定
    )

    valid_loader = DataLoader(
        validLoader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # CPU 训练时 num_workers 设为0更稳定
    )


    # -------------------- 初始化模型 --------------------
    # 扩散过程模型
    eps_model = SimpleDiffusion(
        time_steps=timestep,
        image_shape=(1, image_size, image_size),  # MNIST 单通道
        device=device
    )

    # U-Net 去噪网络
    unet = U_net(
        num_layers=3,  # 下采样5次：32→16→8→4→2→1
        in_ch=1,
        out_ch=1
    ).to(device)

    # 时间嵌入模块
    time_embedding = TimeEmbedding().to(device)

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    ema = EMA(unet, decay=ema_decay)


    # -------------------- 训练循环 --------------------
    print("开始训练...")
    history = []
    valid_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)  # [B, 1, 32, 32]

            batch_size_curr = images.size(0)
            t = torch.randint(0, timestep, (batch_size_curr,), device=device, dtype=torch.long)

            loss = train_one_step(images, t, eps_model, time_embedding, unet, criterion, optimizer)
            ema.update(unet)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        history.append(avg_epoch_loss)

        with torch.no_grad():
            unet.eval()
            valid_loss = 0.0
            for batch_idx, (images, _) in enumerate(valid_loader):
                images = images.to(device)
                batch_size_curr = images.size(0)
                t = torch.randint(0, timestep, (batch_size_curr,), device=device, dtype=torch.long)

                # 前向扩散（与训练完全一致）
                noisy_images, noise = eps_model(images, t)
                t_emb = time_embedding(t)
                predicted_noise = unet(noisy_images, t_emb)

                loss = criterion(predicted_noise, noise)  # 计算预测噪声与真实噪声的 MSE
                valid_loss += loss.item()

            avg_epoch_valid_loss = valid_loss / len(valid_loader)
            valid_history.append(avg_epoch_valid_loss)

        print(f'Epoch {epoch}/{num_epochs} | Train Loss: {avg_epoch_loss:.6f} | Valid Loss: {avg_epoch_valid_loss:.6f}')


        # 保存模型
        if epoch % save_interval == 0 or epoch == num_epochs:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'unet_ema_state_dict': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': history,
                'valid_loss_history': valid_history
            }
            torch.save(checkpoint, model_save_path)
            print(f'模型已保存至 {model_save_path}')

    print("训练结束！")