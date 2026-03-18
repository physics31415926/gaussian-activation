"""
分析 Qwen3 模型结构，找出激活函数位置
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_model_structure(model_name="Qwen/Qwen3-0.6B"):
    """分析模型结构，找出激活函数"""
    
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("\n" + "="*70)
    print("模型结构分析")
    print("="*70)
    
    # 打印模型结构
    print("\n顶层模块:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
    
    # 查找所有激活函数相关的模块
    print("\n\n查找激活函数相关模块:")
    activation_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        # 检查各种可能的激活函数
        if any(act in module_type.lower() for act in ['relu', 'gelu', 'silu', 'swish', 'sigmoid', 'tanh', 'activation', 'act']):
            activation_modules.append((name, module_type, module))
            print(f"  Found: {name} -> {module_type}")
    
    # 查找 MLP 层
    print("\n\n查找 MLP 层:")
    mlp_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'mlp' in module_type.lower():
            mlp_modules.append((name, module_type, module))
            print(f"  Found: {name} -> {module_type}")
            
            # 打印 MLP 子模块
            for sub_name, sub_module in module.named_children():
                print(f"    └── {sub_name}: {type(sub_module).__name__}")
    
    # 检查模型配置中的激活函数
    print("\n\n模型配置:")
    config = model.config
    print(f"  hidden_act: {getattr(config, 'hidden_act', 'N/A')}")
    print(f"  hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  intermediate_size: {getattr(config, 'intermediate_size', 'N/A')}")
    
    # 检查是否有 _activations 或类似的属性
    print("\n\n检查模型内部激活函数定义:")
    for name, module in model.named_modules():
        if hasattr(module, 'ACT2FN'):
            print(f"  {name} has ACT2FN: {module.ACT2FN}")
        if hasattr(module, 'activation_fn'):
            print(f"  {name} has activation_fn: {type(module.activation_fn).__name__}")
        if hasattr(module, 'act_fn'):
            print(f"  {name} has act_fn: {type(module.act_fn).__name__}")
    
    # 深入分析第一个 MLP 模块
    if mlp_modules:
        print("\n\n深入分析第一个 MLP 模块:")
        name, module_type, mlp = mlp_modules[0]
        print(f"MLP: {name}")
        for sub_name, sub_module in mlp.named_children():
            print(f"  {sub_name}: {type(sub_module).__name__}")
            # 检查是否有 callable 属性
            for attr_name in dir(sub_module):
                if not attr_name.startswith('_'):
                    attr = getattr(sub_module, attr_name)
                    if callable(attr) and not isinstance(attr, nn.Module):
                        pass  # print(f"    {attr_name}: {type(attr).__name__}")
    
    return model, activation_modules, mlp_modules


if __name__ == "__main__":
    analyze_model_structure()
