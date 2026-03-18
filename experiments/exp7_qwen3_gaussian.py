"""
实验7: Qwen3-0.6B 激活函数替换实验
将 Qwen3-0.6B 的激活函数替换为 LearnableGaussian
"""
import sys
import os

# 自动处理 Colab 和本地路径
if os.path.exists('/content/gaussian-activation'):
    sys.path.insert(0, '/content/gaussian-activation')
else:
    sys.path.insert(0, '..')

import torch
import torch.nn as nn
import time
import json
from pathlib import Path

# ============================================================
# LearnableGaussian 激活函数
# ============================================================
class LearnableGaussian(nn.Module):
    """
    完全可学习的 Gaussian 激活函数
    
    f(x) = gamma * exp(-(x - mu)^2 / (2 * sigma^2)) + beta
    """
    def __init__(self, init_mu=0.0, init_sigma=1.0, init_gamma=1.0, init_beta=0.0):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu))
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, x):
        sigma = torch.abs(self.sigma) + 1e-8
        gaussian = torch.exp(-((x - self.mu) ** 2) / (2 * sigma ** 2))
        return self.gamma * gaussian + self.beta
    
    def extra_repr(self):
        return f'mu={self.mu.item():.4f}, sigma={self.sigma.item():.4f}, gamma={self.gamma.item():.4f}, beta={self.beta.item():.4f}'


# ============================================================
# 模型加载和激活函数替换
# ============================================================
def load_qwen3_model(model_name="Qwen/Qwen3-0.6B"):
    """加载 Qwen3 模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def replace_activation_with_gaussian(model):
    """
    将模型中的激活函数替换为 LearnableGaussian
    
    Qwen3 使用的激活函数可能是:
    - SiLU (Swish)
    - GELU
    - ReLU
    """
    import torch.nn as nn
    from transformers.activations import ACT2FN
    
    replaced_count = 0
    
    def replace_module(module, name=""):
        nonlocal replaced_count
        
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # 检查是否是激活函数层
            if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh)):
                # 替换为 LearnableGaussian
                setattr(module, child_name, LearnableGaussian())
                replaced_count += 1
                print(f"  Replaced: {full_name} ({type(child).__name__} -> LearnableGaussian)")
            
            # 检查是否是 Activation 类 (transformers 的激活函数)
            elif hasattr(child, '__class__') and child.__class__.__name__ in ['SiLU', 'GELU', 'ReLU', 'QuickGELU']:
                setattr(module, child_name, LearnableGaussian())
                replaced_count += 1
                print(f"  Replaced: {full_name} ({child.__class__.__name__} -> LearnableGaussian)")
            
            # 递归处理子模块
            else:
                replace_module(child, full_name)
    
    print("\nReplacing activation functions...")
    replace_module(model)
    print(f"\nTotal replaced: {replaced_count} activation functions")
    
    return model, replaced_count


def count_parameters(model):
    """统计模型参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model_forward(model, tokenizer, device="cuda"):
    """测试模型前向传播"""
    print("\nTesting forward pass...")
    
    # 简单测试输入
    test_input = "The capital of France is"
    
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Output: {generated_text}")
    
    return generated_text


def analyze_activation_params(model):
    """分析 LearnableGaussian 参数"""
    gaussian_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, LearnableGaussian):
            gaussian_params.append({
                'name': name,
                'mu': module.mu.item(),
                'sigma': module.sigma.item(),
                'gamma': module.gamma.item(),
                'beta': module.beta.item(),
            })
    
    return gaussian_params


def main():
    print("="*70)
    print("实验7: Qwen3-0.6B 激活函数替换实验")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # 检查 GPU 内存
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ============================================================
    # 1. 加载原始模型
    # ============================================================
    print("\n" + "="*70)
    print("1. 加载原始 Qwen3-0.6B 模型")
    print("="*70)
    
    model_orig, tokenizer = load_qwen3_model("Qwen/Qwen3-0.6B")
    total_params, trainable_params = count_parameters(model_orig)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试原始模型
    print("\nTesting original model:")
    orig_output = test_model_forward(model_orig, tokenizer, device)
    
    # ============================================================
    # 2. 替换激活函数
    # ============================================================
    print("\n" + "="*70)
    print("2. 替换激活函数为 LearnableGaussian")
    print("="*70)
    
    # 重新加载模型进行替换
    del model_orig
    torch.cuda.empty_cache()
    
    model_gaussian, tokenizer = load_qwen3_model("Qwen/Qwen3-0.6B")
    model_gaussian, replaced_count = replace_activation_with_gaussian(model_gaussian)
    
    total_params, trainable_params = count_parameters(model_gaussian)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============================================================
    # 3. 测试替换后的模型
    # ============================================================
    print("\n" + "="*70)
    print("3. 测试替换后的模型")
    print("="*70)
    
    gaussian_output = test_model_forward(model_gaussian, tokenizer, device)
    
    # ============================================================
    # 4. 分析激活函数参数
    # ============================================================
    print("\n" + "="*70)
    print("4. 分析 LearnableGaussian 参数")
    print("="*70)
    
    gaussian_params = analyze_activation_params(model_gaussian)
    print(f"Total LearnableGaussian layers: {len(gaussian_params)}")
    
    if len(gaussian_params) > 0:
        print("\nSample parameters (first 5):")
        for i, p in enumerate(gaussian_params[:5]):
            print(f"  {i+1}. {p['name'][:50]}...")
            print(f"     mu={p['mu']:.4f}, sigma={p['sigma']:.4f}, gamma={p['gamma']:.4f}, beta={p['beta']:.4f}")
    
    # ============================================================
    # 5. 对比分析
    # ============================================================
    print("\n" + "="*70)
    print("5. 结果对比")
    print("="*70)
    
    print("\n原始模型输出:")
    print(f"  {orig_output}")
    print("\n替换后模型输出:")
    print(f"  {gaussian_output}")
    
    # 保存结果
    Path('results').mkdir(exist_ok=True)
    results = {
        'model': 'Qwen/Qwen3-0.6B',
        'replaced_activations': replaced_count,
        'total_parameters': total_params,
        'gaussian_layers': len(gaussian_params),
        'original_output': orig_output,
        'gaussian_output': gaussian_output,
        'sample_params': gaussian_params[:10] if gaussian_params else [],
    }
    
    with open('results/exp7_qwen3_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Results saved to: results/exp7_qwen3_results.json")
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    print("\n关键发现:")
    print(f"- 成功替换 {replaced_count} 个激活函数")
    print(f"- 模型参数量: {total_params:,}")
    print(f"- LearnableGaussian 层数: {len(gaussian_params)}")
    print(f"\n注意: 替换后的模型输出可能与原始模型不同，需要进一步微调。")


if __name__ == "__main__":
    main()
