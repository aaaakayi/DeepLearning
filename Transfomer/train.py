import sys
import os

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.encoder_decoder import Transformer
from Dataset.vocab.vocab_build import get_vocab, VocabularyBuilder
from Dataset.dataset import get_train_data, TranslationDataset
from torch.utils.data import DataLoader

# 导入nltk用于BLEU计算
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 下载必要的数据（如果需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# 创建一个虚拟的 'vocab' 模块，指向实际的 Dataset.vocab
import Dataset.vocab.vocab_build as vocab_module

sys.modules['vocab'] = vocab_module
sys.modules['vocab.vocab_build'] = vocab_module


def collate_fn_train(batch, pad_token_id=0):
    """训练用的批处理函数，返回字典格式"""
    src_max_len = max(len(item['src']) for item in batch)
    tgt_max_len = max(len(item['tgt']) for item in batch)

    batch_size = len(batch)
    src_batch = torch.full((batch_size, src_max_len), pad_token_id, dtype=torch.long)
    tgt_batch = torch.full((batch_size, tgt_max_len), pad_token_id, dtype=torch.long)

    src_texts = []
    tgt_texts = []

    for i, item in enumerate(batch):
        src_len = len(item['src'])
        tgt_len = len(item['tgt'])

        src_batch[i, :src_len] = item['src']
        tgt_batch[i, :tgt_len] = item['tgt']

        # 保存文本（用于调试）
        if 'src_text' in item:
            src_texts.append(item['src_text'])
        if 'tgt_text' in item:
            tgt_texts.append(item['tgt_text'])


    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }


def evaluate(model, dataloader, criterion, vocab_src, vocab_tgt, device, max_len=50):
    """
    Transformer模型的评估函数
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    # 获取特殊token的id
    pad_idx = vocab_tgt.word2idx.get('<pad>', 0)
    bos_idx = vocab_tgt.word2idx.get('<bos>', 1)
    eos_idx = vocab_tgt.word2idx.get('<eos>', 2)

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # === 计算损失 ===
            # decoder输入: tgt去掉最后一个token(EOS)
            # 注意：tgt包含BOS和EOS
            logits, _ = model(src_seq=src, tgt_seq=tgt[:, :-1])

            # 计算损失
            # 目标: tgt去掉第一个token(BOS)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()

            # === 生成翻译（用于BLEU）===
            generated = model.generate(
                src_seq=src,
                max_len=max_len,
                temperature=1.0,
                bos_token_id=bos_idx,
                eos_token_id=eos_idx
            )

            # 处理每个样本
            for i in range(src.size(0)):
                # 获取生成的token ids
                gen_ids = generated[i].cpu().tolist()

                # 去掉BOS（如果存在）
                if len(gen_ids) > 0 and gen_ids[0] == bos_idx:
                    gen_ids = gen_ids[1:]

                # 去掉EOS及其之后的部分
                if eos_idx in gen_ids:
                    eos_pos = gen_ids.index(eos_idx)
                    gen_ids = gen_ids[:eos_pos]

                # 获取目标序列
                target_ids = tgt[i].cpu().tolist()
                # 去掉BOS
                if len(target_ids) > 0 and target_ids[0] == bos_idx:
                    target_ids = target_ids[1:]
                # 去掉EOS
                if len(target_ids) > 0 and target_ids[-1] == eos_idx:
                    target_ids = target_ids[:-1]

                # 转换为单词列表
                def ids_to_words(indices):
                    words = []
                    for idx in indices:
                        if idx == pad_idx:  # 跳过pad
                            continue
                        word = vocab_tgt.idx2word.get(idx, f'[UNK{idx}]')
                        words.append(word)
                    return words

                pred_words = ids_to_words(gen_ids)
                target_words = ids_to_words(target_ids)

                # 跳过空句子
                if not pred_words or not target_words:
                    continue

                all_predictions.append(pred_words)
                all_targets.append([target_words])

    # 计算BLEU（使用nltk）
    if all_predictions and all_targets:
        # 使用平滑函数处理短句
        smoothing_function = SmoothingFunction().method1
        bleu = corpus_bleu(all_targets, all_predictions,
                           smoothing_function=smoothing_function)
        bleu_score = bleu * 100  # 转换为百分比
    else:
        bleu_score = 0.0

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    return {
        'loss': avg_loss,
        'bleu': bleu_score,
        'predictions': all_predictions,
        'targets': all_targets
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        optimizer.zero_grad()

        # decoder输入: tgt去掉最后一个token(EOS)
        logits, _ = model(src_seq=src, tgt_seq=tgt[:, :-1])

        # 计算损失，目标: tgt去掉第一个token(BOS)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_with_evaluation(model, train_loader, valid_loader, optimizer, criterion,
                          vocab_src, vocab_tgt, device, num_epochs, save_path):
    """
    训练并在验证集上评估
    """
    train_losses = []
    valid_metrics = []
    best_bleu = 0

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 使用tqdm创建epoch进度条
    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", unit="epoch")

    for epoch in epoch_pbar:
        # === 训练 ===
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(avg_train_loss)

        # === 验证 ===
        valid_results = evaluate(
            model, valid_loader, criterion,
            vocab_src, vocab_tgt, device
        )

        valid_metrics.append(valid_results)

        # 更新进度条描述
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_pbar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'valid_loss': f"{valid_results['loss']:.4f}",
            'valid_bleu': f"{valid_results['bleu']:.4f}"
        })

        # 保存最佳模型
        if valid_results['bleu'] > best_bleu:
            best_bleu = valid_results['bleu']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_results['loss'],
                'bleu': best_bleu,
            }, os.path.join(save_path, "best_model.pt"))
            epoch_pbar.write(f"  ✓ 保存最佳模型 (BLEU: {best_bleu:.4f})")

        epoch_pbar.write("-" * 50)

    return {
        'train_losses': train_losses,
        'valid_metrics': valid_metrics
    }


if __name__ == "__main__":
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")

    # 获取字典
    vocab_zh, vocab_en = get_vocab()
    print(f"英文词汇表大小: {len(vocab_en)}")
    print(f"中文词汇表大小: {len(vocab_zh)}")

    # 获取pad_token_id
    pad_token_id = vocab_en.word2idx.get('<pad>', 0)
    print(f"Pad token ID: {pad_token_id}")

    # 构建训练集
    print("\n构建训练集...")
    train_dataset = TranslationDataset(
        src_file=r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.en",
        tgt_file=r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_train_en-zh.tok.zh",
        src_vocab=vocab_en,
        tgt_vocab=vocab_zh,
        max_len=50,
    )
    print(f"训练集大小: {len(train_dataset)}")

    # 验证集
    print("构建验证集...")
    valid_dataset = TranslationDataset(
        src_file=r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_dev_en-zh.tok.en",
        tgt_file=r"C:/Users/Aumu/PycharmProjects/DeepLearning/data/en-zh/tok/ted_dev_en-zh.tok.zh",
        src_vocab=vocab_en,
        tgt_vocab=vocab_zh,
        max_len=50
    )
    print(f"验证集大小: {len(valid_dataset)}")

    # 模型
    print("\n初始化模型...")
    model = Transformer(
        src_vocab_size=len(vocab_en),
        tgt_vocab_size=len(vocab_zh),
        num_layers=5,
        d_model=256,
        num_heads=8,
        dropout=0.1,
        max_src_len=50,
        max_tgt_len=50,
    )
    model = model.to(device=device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_train(batch, pad_token_id=pad_token_id)
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_train(batch, pad_token_id=pad_token_id)
    )

    # 损失函数（忽略pad_token）
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用较小的学习率

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 开始训练
    num_epochs = 5
    save_path = "./checkpoint"

    print(f"\n开始训练，共 {num_epochs} 个epoch...")

    history = train_with_evaluation(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        vocab_src=vocab_en,
        vocab_tgt=vocab_zh,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path
    )

    print("\n训练完成!")
    print(f"训练损失历史: {history['train_losses']}")

    if history['valid_metrics']:
        bleu_history = [m['bleu'] for m in history['valid_metrics']]
        print(f"验证BLEU历史: {bleu_history}")
        print(f"最佳BLEU: {max(bleu_history):.4f}")