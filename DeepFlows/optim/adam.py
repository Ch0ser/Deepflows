from .optimier import Optimizer
# 假设你提供的 backendtensor 文件在当前包或正确导入
# 如果是在 backend_api.py 中，则 import 对应的 zeros_like
# 这里假设 zeros_like 可以在当前上下文访问，或者你需要从你的 backend 模块导入它
from .. import backend_api 

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay=0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 【修正1】使用 zeros_like 初始化，更安全
        # 注意：这里我们需要访问 param 的底层 data 来创建 zero tensor
        self.v = [backend_api.zeros_like(param.data) for param in self.params]
        self.s = [backend_api.zeros_like(param.data) for param in self.params]
        self.t = 1

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # 【修正2】获取纯数据 (BackendTensor)，切断计算图
            # 如果 param.grad 是 Tensor (有 .data)，取 .data
            # 如果 param.grad 已经是 BackendTensor (无 .data)，直接用
            grad_data = param.grad.data if hasattr(param.grad, 'data') else param.grad
            
            # 获取参数的纯数据
            p_data = param.data if hasattr(param, 'data') else param

            # 权重衰减 (全在 BackendTensor 层面计算)
            if self.weight_decay > 0:
                grad_data = grad_data + p_data * self.weight_decay

            # 【修正3】动量更新 (纯数值计算，不产生 Autograd 节点)
            # self.v[i] 和 grad_data 都是 BackendTensor，运算结果也是新的 BackendTensor
            self.v[i] = self.v[i] * self.beta1 + grad_data * (1 - self.beta1)
            self.s[i] = self.s[i] * self.beta2 + grad_data**2 * (1 - self.beta2)

            # 偏差修正
            v_hat = self.v[i] / (1 - self.beta1 ** self.t)
            s_hat = self.s[i] / (1 - self.beta2 ** self.t)

            # 计算更新量
            update = v_hat / (s_hat**0.5 + self.eps) * self.lr

            # 【修正4】更新参数
            # 因为 BackendTensor 没有 __isub__，我们显式计算并赋值
            # 这样 param.data 指向更新后的新 BackendTensor
            param.data = p_data - update
            param.children.clear() 
            param.parents.clear() # 作为一个好习惯，把上游也清了（虽然权重通常没parent）
            
        self.t += 1