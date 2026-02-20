# U-net 被用于去噪过程中,模拟一个去噪过程
# U-net 的组成 : 收缩路径(encoder), 瓶颈层（Bottleneck)连接编码器和解码器 ,扩张路径(decoder)

import torch
import torch.nn as nn
class TimeEmbedding(nn.Module):
    """
        Sinusoidal Positional Encoding
        :param t: (batch_size,)
        :param d_model: 嵌入维度
        :return batch_size , d_model
    """
    def __init__(self,d_model = 256):
        super().__init__()
        self.d_model = d_model

    def forward(self,t):
        if not isinstance(t,torch.Tensor):
            t = torch.tensor([t],dtype=torch.float32)
        else:
            t = t.float()

        device = t.device
        batch_size = t.shape[0]


        #emb_freq = 1/(10000**((2i)/d)) = exp [-(2i)/d * ln10000] = exp [i * -ln10000/(d/2) ]
        half_dim = self.d_model // 2
        embed_factor = torch.log(torch.tensor(10000.0,device=device)) / (half_dim - 1) # ln10000/(d/2),half_dim - 1作为d/2用于工程上考量
        embed_freq = torch.exp(torch.arange(half_dim,device=device) * -embed_factor) # exp[i * -embed_factor]

        # Compute angles: t * frequency factor
        angles = t.unsqueeze(-1) * embed_freq.unsqueeze(0)  # (batch, half_dim)

        embedding = torch.zeros((batch_size,self.d_model),device=device)
        embedding[:, 0::2] = torch.sin(angles)  # even indices
        embedding[:, 1::2] = torch.cos(angles)  # odd indices

        return embedding


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim = 256):
        super().__init__()

        # 1. 定义两个卷积层
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1)

        # 2. 定义组归一化
        max_groups = 32
        # 组数不能超过通道数，且必须能被通道数整除
        num_groups = min(max_groups, out_ch)
        # 找到不大于 num_groups 且能整除 out_ch 的最大整数
        while out_ch % num_groups != 0:
            num_groups -= 1

        self.norm1 = nn.GroupNorm(num_groups=num_groups,num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)

        # 3. 定义时间嵌入映射层 (Linear: time_emb_dim → out_ch*2)
        self.time_mlp = nn.Linear(time_emb_dim,out_ch*2)

        # 4. 残差连接：若 in_ch != out_ch，需用1x1卷积调整
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        self.activation = nn.SiLU()

    def forward(self, x, t_emb):

        # 获取残差连接的值
        skip = self.skip(x)

        # 第一层卷积层与组归一化
        h = self.conv1(x)
        h = self.norm1(h)

        # ----- 从时间嵌入生成 gamma 和 beta -----
        params = self.time_mlp(t_emb)
        # 沿通道维拆分成 gamma 和 beta
        gamma, beta = params.chunk(2, dim=-1)
        # 增加空间维度便于广播
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]  # [batch, channel, 1, 1]

        # ----- AdaGN: gamma * norm(h) + beta -----
        h = gamma * h + beta

        # ----- 激活 -----
        h = self.activation(h)

        # ----- 第2卷积 -----
        h = self.conv2(h)  # [2,4,32,32]
        h = self.norm2(h)  # [2,4,32,32]

        # ----- 再次应用 AdaGN（可用同一组 gamma, beta，也可学习另一组）-----
        # 简单起见，复用同一个时间映射，得到新参数或共享均可
        h = gamma * h + beta

        # ----- 残差连接 + 最终激活 -----
        h = h + skip
        h = self.activation(h)

        return h
    
class encoder(nn.Module):
    def __init__(self,num_layers,in_ch,out_ch):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(ConvBlock(in_ch=in_ch,out_ch=out_ch))

            if i < self.num_layers-1:
                self.down.append(
                    nn.Conv2d(
                        in_channels=out_ch,
                        out_channels=2 * out_ch,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    )
                )

            # 每一层的通道数都要更新
            in_ch = 2 * out_ch
            out_ch = in_ch  # 一般把这两个设计成一致的大小

    def forward(self,x,t_emb):
        residual = []
        for i,blk in enumerate(self.blocks):
            x = blk(x,t_emb)
            if i < self.num_layers - 1:
                residual.append(x)
                x = self.down[i](x)

        return x,residual

class decoder(nn.Module):
    def __init__(self,num_layers,in_ch,out_ch):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.up = nn.ModuleList()
        self.num_layers = num_layers
        self.channels = [in_ch//(2**i) for i in range(num_layers)]
        for i in range(num_layers):
            if i == 0:
                block_in = self.channels[i]
                block_out = self.channels[i]
            else:
                block_in = self.channels[i] * 2
                block_out = self.channels[i]

            self.blocks.append(ConvBlock(in_ch=block_in,out_ch=block_out))

            if i < num_layers - 1:
                self.up.append(
                    nn.ConvTranspose2d(
                        in_channels = block_out,
                        out_channels = block_out // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    )
                    # 通道数减半,像素翻倍
                    # output_size = (input_size - 1) * stride + kernel_size - 2 * padding + output_padding
                )

        self.final_conv = nn.Conv2d(
            in_channels=self.channels[-1],
            out_channels=out_ch,
            kernel_size=3,
            padding=1
        )

    def forward(self,x,t_emb,residual):
        for i,blk in enumerate(self.blocks):
            if i > 0:
                #其余层需要跳跃连接
                skip = residual[-(i)]
                x = torch.cat([x, skip], dim=1) #沿着通道数连接 batch_size, channels, H, W

            x = blk(x=x,t_emb=t_emb)

            if i < self.num_layers - 1:
                x = self.up[i](x)
        x = self.final_conv(x)

        return x

class U_net(nn.Module):
    def __init__(self,num_layers,in_ch,out_ch):
        super().__init__()
        self.encoder = encoder(num_layers,in_ch,out_ch)
        self.decoder = decoder(
            num_layers,
            in_ch * 2 ** (num_layers - 1),
            out_ch
        )

    def forward(self,x,t_emb):
        x,residual = self.encoder(x,t_emb)
        x = self.decoder(x,t_emb,residual)

        return x



def load_image_pil_torch(image_path):
    from PIL import Image
    import torch
    from torchvision import transforms
    # 打开图片
    image = Image.open(image_path)
    image = Image.open(image_path).convert('RGB')

    # 定义转换管道
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整大小
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量 [0,1]范围
        transforms.Normalize(  # 标准化（常用ImageNet统计量）
            mean=[0.485, 0.456, 0.406],  # RGB均值
            std=[0.229, 0.224, 0.225]  # RGB标准差
        )
    ])

    # 应用转换
    tensor = transform(image)  # 形状: [C, H, W]

    #添加通道数
    tensor = torch.unsqueeze(tensor,0)

    return tensor

def draw(data):
    """
    :param data: list of image tensors, each tensor shape (batch, C, H, W) or (C, H, W)
    """
    import matplotlib.pyplot as plt
    pic_num = len(data)

    fig, axes = plt.subplots(1, pic_num)
    if pic_num == 1:
        axes = [axes]

    for i, x in enumerate(data):
        if x.dim() == 4:
            img = x[0]
        else:
            img = x

        # 转换为 numpy 并调整维度顺序 (H, W, C)
        img_np = img.cpu().permute(1, 2, 0).detach().numpy()

        axes[i].imshow(img_np)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    d_model = 256
    T_embed_model = TimeEmbedding(d_model)
    t = 10
    t_emb = T_embed_model(t)
    pic = []


    x = load_image_pil_torch('Strawberry.png')
    pic.append(x) # 加入原图
    print(f"x shape:{x.shape}")

    model = U_net(num_layers=5,in_ch=3,out_ch=3)
    x = model(x,t_emb)
    pic.append(x)

    draw(pic)



