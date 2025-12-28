import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ..nn.modules.module import Module
from ..optim.optimier import Optimizer
from ..tensor import Tensor
from .. import backend_api

# 尝试导入dill库，若不可用则使用pickle
try:
    import dill as pickle
    print("使用dill库进行序列化，支持更多Python对象类型")
except ImportError:
    import pickle
    print("警告: dill库不可用，使用pickle替代。部分复杂对象可能无法序列化。")


def save_checkpoint(model: Module,
                    optimizer: Optional[Optimizer] = None,
                    epoch: int = 0,
                    loss: Optional[float] = None,
                    save_path: str = 'checkpoint.pkl') -> None:
    """
    保存模型的完整训练状态，简化版实现
    
    Args:
        model: 要保存的模型
        optimizer: 优化器实例
        epoch: 当前训练轮次
        loss: 当前损失值
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 创建检查点字典，只保存最基本的数据
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_parameters': {}
    }
    
    # 保存模型参数 - 只保存numpy数组
    for name, param in model.named_parameters():
        try:
            # 直接保存numpy数组
            if hasattr(param.data, 'numpy'):
                checkpoint['model_parameters'][name] = param.data.numpy()
            elif isinstance(param.data, np.ndarray):
                checkpoint['model_parameters'][name] = param.data.copy()
        except Exception as e:
            print(f"警告: 无法保存参数 {name}: {e}")
    
    # 保存优化器状态 
    if optimizer is not None:
        optimizer_state = {
            'type': type(optimizer).__name__
        }
        
        # 保存基本超参数
        for attr in ['lr', 'momentum', 'weight_decay']:
            if hasattr(optimizer, attr):
                optimizer_state[attr] = getattr(optimizer, attr)
        
        # 保存Adam特有的内部状态
        if hasattr(optimizer, 'v'):
            optimizer_state['v'] = [v.data.numpy() if hasattr(v, 'data') else v.numpy() for v in optimizer.v]
        if hasattr(optimizer, 's'):
            optimizer_state['s'] = [s.data.numpy() if hasattr(s, 'data') else s.numpy() for s in optimizer.s]
        if hasattr(optimizer, 't'):
            optimizer_state['t'] = optimizer.t
        
        checkpoint['optimizer_state'] = optimizer_state
    
    # 保存检查点
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"检查点已保存到 {save_path}")
    except Exception as e:
        print(f"保存检查点失败: {str(e)}")


def load_checkpoint(model: Module,
                    optimizer: Optional[Optimizer] = None,
                    save_path: str = 'checkpoint.pkl') -> Dict[str, Any]:
    """
    加载模型的完整训练状态
    
    Args:
        model: 要恢复的模型
        optimizer: 优化器实例
        save_path: 检查点文件路径
    
    Returns:
        包含训练状态信息的字典
    """
    # 检查文件是否存在
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"检查点文件不存在: {save_path}")
    
    # 加载检查点
    try:
        with open(save_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"加载检查点失败: {str(e)}")
    
    # 恢复模型参数
    if 'model_parameters' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['model_parameters']:
                param_data = checkpoint['model_parameters'][name]
                try:
                    # 尝试使用更安全的方式恢复参数数据
                    if isinstance(param_data, np.ndarray) and hasattr(param, 'data'):
                        # 检查参数是否有device属性
                        device = param.device if hasattr(param, 'device') else default_device()
                        # 直接创建一个新的BackendTensor对象来替换数据
                        if hasattr(backend_api, 'Btensor'):
                            # 使用backend_api来创建适合设备的张量
                            param.data = backend_api.Btensor(param_data, dtype='float32', device=device)
                        else:
                            # 如果没有Btensor，则尝试直接设置
                            param.data._data = param_data
                except Exception as e:
                    print(f"警告: 无法恢复参数 {name}: {str(e)}")
    
    # 恢复优化器状态
    if optimizer is not None and 'optimizer_state' in checkpoint:
        opt_state = checkpoint['optimizer_state']
        
        # 恢复基本超参数
        for attr in ['lr', 'momentum', 'weight_decay']:
            if attr in opt_state and hasattr(optimizer, attr):
                setattr(optimizer, attr, opt_state[attr])
        
        # 恢复Adam特有的内部状态
        if hasattr(optimizer, 'v') and 'v' in opt_state:
            # 直接从opt_state中恢复v状态
            for i, param in enumerate(optimizer.params):
                if i < len(opt_state['v']):
                    np_array = opt_state['v'][i]
                    if isinstance(np_array, np.ndarray):
                        # 检查参数是否有device属性
                        device = param.device if hasattr(param, 'device') else default_device()
                        # 直接创建一个与参数类型匹配的BackendTensor
                        if hasattr(backend_api, 'Btensor'):
                            # 使用backend_api创建适合设备的张量
                            v_tensor = backend_api.Btensor(np_array, dtype='float32', device=device)
                            optimizer.v[i] = v_tensor
                        else:
                            # 如果没有Btensor，则使用Tensor对象
                            optimizer.v[i] = Tensor(np_array, device=device, requires_grad=False)
        if hasattr(optimizer, 's') and 's' in opt_state:
            # 直接从opt_state中恢复s状态
            for i, param in enumerate(optimizer.params):
                if i < len(opt_state['s']):
                    np_array = opt_state['s'][i]
                    if isinstance(np_array, np.ndarray):
                        # 检查参数是否有device属性
                        device = param.device if hasattr(param, 'device') else default_device()
                        # 直接创建一个与参数类型匹配的BackendTensor
                        if hasattr(backend_api, 'Btensor'):
                            s_tensor = backend_api.Btensor(np_array, dtype='float32', device=device)
                            optimizer.s[i] = s_tensor
                        else:
                            # 如果没有Btensor，则使用Tensor对象
                            optimizer.s[i] = Tensor(np_array, device=device, requires_grad=False)
        if hasattr(optimizer, 't') and 't' in opt_state:
            # 直接从opt_state中恢复t状态
            optimizer.t = opt_state['t']
    
    print(f"检查点已从 {save_path} 加载")
    
    # 返回训练状态信息
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss')
    }


