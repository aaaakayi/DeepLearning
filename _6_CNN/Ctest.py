import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def diagnose_model_issues(model, X_sample, y_sample):
    """诊断模型问题"""
    print("=" * 60)
    print("模型诊断报告")
    print("=" * 60)

    # 1. 检查输入数据
    print("\n1. 输入数据检查:")
    print(f"  输入形状: {X_sample.shape}")
    print(f"  数据范围: [{X_sample.min():.3f}, {X_sample.max():.3f}]")
    print(f"  均值: {X_sample.mean():.3f}, 标准差: {X_sample.std():.3f}")
    print(f"  标签: {y_sample}")

    # 2. 逐层检查输出
    print("\n2. 逐层输出检查:")
    layer_outputs = []
    layer_names = []

    # 获取中间层输出
    for i, layer in enumerate(model.layers):
        # 创建子模型，输出到当前层
        intermediate_model = tf.keras.Model(
            inputs=model.input,
            outputs=layer.output
        )

        # 前向传播到当前层
        intermediate_output = intermediate_model.predict(X_sample, verbose=0)
        layer_outputs.append(intermediate_output)
        layer_names.append(layer.name)

        print(f"  第{i + 1}层: {layer.name}")
        print(f"    输出形状: {intermediate_output.shape}")

        # 检查是否全零或全相同
        if len(intermediate_output.shape) > 1:
            flat_output = intermediate_output.flatten()
            unique_values = np.unique(flat_output[:100])  # 检查前100个值
            if len(unique_values) <= 3:
                print(f"    ⚠️  警告: 输出值过于单一，仅 {len(unique_values)} 个不同值")
                if np.all(flat_output == 0):
                    print(f"    ❌ 严重: 输出全为零!")

            # 检查值范围
            print(f"    值范围: [{intermediate_output.min():.6f}, {intermediate_output.max():.6f}]")

            # 检查NaN或Inf
            if np.any(np.isnan(intermediate_output)):
                print(f"    ❌ 严重: 包含NaN值!")
            if np.any(np.isinf(intermediate_output)):
                print(f"    ❌ 严重: 包含Inf值!")

    # 3. 检查梯度
    print("\n3. 梯度检查:")
    with tf.GradientTape() as tape:
        predictions = model(X_sample)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_sample, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            print(f"  参数 {var.name[:30]:30} 梯度范数: {grad_norm:.6f}")

            if grad_norm < 1e-10:
                print(f"    ⚠️  警告: 梯度消失 (norm={grad_norm:.6e})")
            elif grad_norm > 100:
                print(f"    ⚠️  警告: 梯度爆炸 (norm={grad_norm:.6f})")
        else:
            print(f"  参数 {var.name[:30]:30} 无梯度")

    # 4. 预测检查
    print("\n4. 预测检查:")
    predictions = model.predict(X_sample, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)

    print(f"  预测概率形状: {predictions.shape}")
    print(f"  预测标签: {pred_labels}")
    print(f"  真实标签: {y_sample}")

    # 检查预测是否均匀
    avg_probs = predictions.mean(axis=0)
    print(f"  各类别平均概率: {np.round(avg_probs, 4)}")

    if np.allclose(avg_probs, 1.0 / num_classes, atol=0.05):
        print(f"  ⚠️  警告: 预测接近均匀分布 ({1.0 / num_classes:.3f})")

    return layer_outputs, layer_names, gradients


def visualize_layer_outputs(layer_outputs, layer_names):
    """可视化各层输出"""
    n_layers = len(layer_outputs)

    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i, (output, name, ax) in enumerate(zip(layer_outputs, layer_names, axes)):
        if i < len(layer_outputs):
            # 展平输出以便可视化
            if len(output.shape) > 1:
                flat_output = output.flatten()
                # 只取前1000个点
                sample_size = min(1000, len(flat_output))
                sampled = flat_output[:sample_size]

                ax.hist(sampled, bins=50, alpha=0.7)
                ax.set_title(f'{name}\n形状: {output.shape}')
                ax.set_xlabel('激活值')
                ax.set_ylabel('频数')

                # 标记零值比例
                zero_ratio = np.mean(np.abs(sampled) < 1e-10)
                if zero_ratio > 0.5:
                    ax.text(0.05, 0.95, f'零值比例: {zero_ratio:.1%}',
                            transform=ax.transAxes, color='red',
                            verticalalignment='top')

                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()