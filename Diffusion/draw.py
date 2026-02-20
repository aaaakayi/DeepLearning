import matplotlib.pyplot as plt
import torch


def plot_loss_from_checkpoint(checkpoint_path):
    """
    从保存的checkpoint文件中加载损失历史并绘制训练和验证损失曲线。

    参数:
        checkpoint_path: .pth文件的路径
    """
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取损失历史
    train_losses = checkpoint.get('loss_history', [])
    valid_losses = checkpoint.get('valid_loss_history', [])

    if not train_losses and not valid_losses:
        print("Checkpoint中未找到损失历史记录。")
        return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    if valid_losses:
        # 验证损失可能长度与训练相同或略少
        val_epochs = range(1, len(valid_losses) + 1)
        plt.plot(val_epochs, valid_losses, label='Validation Loss', marker='s')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# 使用示例
plot_loss_from_checkpoint('checkpoint/diffusion_mnist.pth')