import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
import numpy as np

from collections import OrderedDict
from typing import Dict, List, Optional

from .problem import DiscreteTestProblem

class DiscreteBranin(DiscreteTestProblem):
    """
    Discrete version of the Branin test function with variable dimensions.
    For dimensions > 2, the function is repeated for each pair of variables.
    
    Domain:
        x[2i] ∈ [-5, 10] (integer), x[2i+1] ∈ [0, 15] (integer) for i = 0,1,...,⌊dim/2⌋-1
    """
    
    # 预定义类变量
    dim = 2
    num_objectives = 1
    
    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = dim
        if dim < 2:
            raise ValueError("Dimension must be at least 2")
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")
            
        # # 设置_bounds
        # bounds = torch.zeros(2, dim)
        # for i in range(dim // 2):
        #     bounds[0, 2*i] = -5
        #     bounds[1, 2*i] = 10
        #     bounds[0, 2*i+1] = 0
        #     bounds[1, 2*i+1] = 15
        
        # self._bounds = bounds

        # 设置_bounds
        bounds = torch.zeros(dim, 2)
        for i in range(dim // 2):
            bounds[2*i, 0] = -5    # 第2i维的下限
            bounds[2*i, 1] = 10    # 第2i维的上限
            bounds[2*i+1, 0] = 0   # 第2i+1维的下限
            bounds[2*i+1, 1] = 15  # 第2i+1维的上限

        self._bounds = bounds

        print(f"Branin problem initialized with bounds: {self._bounds.shape}")
        # 调用父类初始化
        super().__init__(
            noise_std=noise_std,
            negate=negate,
            integer_indices=list(range(dim)),  # 所有维度都是整数值
            categorical_indices=None,
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        pairs = self.dim // 2  # Number of complete pairs
        
        for i in range(pairs):
            x1, x2 = X[..., 2*i], X[..., 2*i + 1]
            a = 1
            b = 5.1 / (4 * torch.pi**2)
            c = 5 / torch.pi
            r = 6
            s = 10
            t = 1 / (8 * torch.pi)
            
            term = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
            result += term
        
        return result

class DiscretePestControl(DiscreteTestProblem):
    """
    Discrete Pest Control problem.
    All variables are integers in [0, 4] representing different control strategies.
    """
    
    # 预定义类变量
    dim = 25
    num_objectives = 1
    
    def __init__(
        self,
        dim: int = 25,
        noise_std: Optional[float] = None,
        negate: bool = False,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        if dim < 25:
            raise ValueError("Dimension must be at least 25")
        
        # 设置_bounds
        bounds = torch.zeros(dim,2)
        bounds[:,1] = 4  # 所有变量范围从0到4
        
        self._bounds = bounds
        self.seed = seed
        
        # 调用父类初始化
        super().__init__(
            noise_std=noise_std,
            negate=negate,
            integer_indices=list(range(dim)),  # 所有维度都是整数值
            categorical_indices=None,
        )
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        # 转为numpy进行评估
        X_np = X.detach().cpu().numpy()
        results = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
        
        # 处理每个批次元素
        for i in range(X.shape[0]):
            x_i = X_np[i]
            # 调用害虫控制函数
            result = self._evaluate_pest_control(x_i, self.seed)
            results[i] = result
            
        return results
    
    def _evaluate_pest_control(self, x, seed=None):
        """
        简化版的害虫控制评估实现
        这只是一个占位符 - 您需要集成实际的pest_control函数
        """
        # 模拟常量
        U = 0.1  # 害虫比例阈值
        n_stages = len(x)
        n_simulations = 100
        
        # 控制措施参数
        control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
        base_tolerance = {1: 1.0/7.0, 2: 2.5/7.0, 3: 2.0/7.0, 4: 0.5/7.0}
        tolerance_develop_rate = {k: v * (25.0 / n_stages) for k, v in base_tolerance.items()}
        
        control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
        control_beta = {1: 2.0/7.0, 2: 3.0/7.0, 3: 3.0/7.0, 4: 5.0/7.0}
        
        # 初始化随机状态
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        # 跟踪成本和风险
        total_cost = 0
        total_risk = 0
        
        # 初始化害虫种群
        init_pest_frac_alpha = 1.0
        init_pest_frac_beta = 30.0
        curr_pest_frac = rng.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
        
        # 通过各阶段模拟
        for i in range(n_stages):
            # 生成传播率
            spread_alpha = 1.0
            spread_beta = 17.0 / 3.0
            spread_rate = rng.beta(spread_alpha, spread_beta, size=(n_simulations,))
            
            # 如果选择了控制措施则应用
            control_choice = int(x[i])
            if control_choice > 0:
                # 生成控制率
                control_alpha = 1.0
                control_rate = rng.beta(control_alpha, control_beta[control_choice], size=(n_simulations,))
                
                # 更新害虫种群
                next_pest_frac = (1.0 - control_rate) * curr_pest_frac
                curr_pest_frac = next_pest_frac
                
                # 更新控制有效性（耐受性发展）
                control_beta[control_choice] += tolerance_develop_rate[control_choice] / float(n_stages)
                
                # 计算带有批量折扣的成本
                discount = control_price_max_discount[control_choice] / float(n_stages) * float(np.sum(x == control_choice))
                stage_cost = control_price[control_choice] * (1.0 - discount)
                total_cost += stage_cost
            else:
                # 未应用控制
                next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
                curr_pest_frac = next_pest_frac
            
            # 如果害虫比例超过阈值，增加风险
            total_risk += np.mean(curr_pest_frac > U)
        
        return total_cost + total_risk