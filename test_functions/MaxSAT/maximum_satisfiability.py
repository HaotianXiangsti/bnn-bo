import os
import numpy as np

# # 打印当前工作目录，帮助调试
# print("当前工作目录:", os.getcwd())

# import sys

# # 打印当前工作目录
# print("当前工作目录:", os.getcwd())

# # 关键步骤：确保项目根目录在Python路径中
# # 获取脚本文件的绝对路径
script_path = os.path.abspath(__file__)
print("脚本路径:", script_path)

# # 计算相对于脚本文件的项目根目录
# # 从 /home/.../LLMBO_NIPS/BOOM/test_funcs/MaxSAT/maximum_satisfiability.py
# # 向上三级到 /home/.../LLMBO_NIPS
# project_root = os.path.abspath(os.path.join(os.path.dirname(script_path), '../../..'))
# print("项目根目录:", project_root)

# # 将项目根目录添加到Python路径的最前面
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#     print(f"已添加 {project_root} 到Python路径")

# # 现在可以导入BOOM模块了
# from BOOM.test_funcs.base import TestFunction
# print("成功导入 TestFunction 模块")

# # MaxSAT数据目录路径
# MAXSAT_DIR_NAME = os.path.join(os.path.dirname(script_path), 'maxsat2018_data')
# print("MAXSAT数据目录:", MAXSAT_DIR_NAME)

# # 检查数据目录是否存在
# if not os.path.exists(MAXSAT_DIR_NAME):
#     print(f"创建 MaxSAT 数据目录: {MAXSAT_DIR_NAME}")
#     os.makedirs(MAXSAT_DIR_NAME, exist_ok=True)
    

import os
import sys
import numpy as np
import torch
import shutil
import tarfile
import requests
from pathlib import Path

# MaxSAT数据目录路径
MAXSAT_DIR_NAME = os.path.join(os.path.dirname(script_path), 'maxsat2018_data')
os.makedirs(MAXSAT_DIR_NAME, exist_ok=True)

# 下载MaxSAT数据文件
def download_maxsat60_data():
    """下载并准备MaxSAT60数据文件"""
    data_file = os.path.join(MAXSAT_DIR_NAME, 'frb-frb10-6-4.wcnf')
    
    if os.path.exists(data_file):
        print(f"数据文件已存在: {data_file}")
        return
    
    print(f"开始下载MaxSAT60数据...")
    url = "http://bounce-resources.s3-website-us-east-1.amazonaws.com/wms_crafted.tgz"
    
    try:
        # 创建临时目录
        temp_dir = os.path.join(MAXSAT_DIR_NAME, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 下载文件
        response = requests.get(url, verify=False)
        tgz_path = os.path.join(temp_dir, "wms_crafted.tgz")
        with open(tgz_path, "wb") as file:
            file.write(response.content)
        
        # 解压文件
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(temp_dir)
        
        # 移动数据文件到正确位置
        src_path = os.path.join(temp_dir, "wms_crafted", "frb", "frb10-6-4.wcnf")
        shutil.copy(src_path, data_file)
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"MaxSAT60数据已下载并解压到: {data_file}")
    
    except Exception as e:
        print(f"下载数据时发生错误: {e}")
        raise


# An abstract class implementation for all test functions

from abc import abstractmethod
import numpy as np


class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, 'int_constrained_dims must be a subset of the continuous_dims, ' \
                                                 'but continuous_dims is not supplied!'
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), "all continuous dimensions with integer " \
                                                           "constraint must be themselves contained in the " \
                                                           "continuous_dimensions!"

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x = np.array([np.random.choice(self.config[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False, ))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
import torch



# from COMBO.experiments.exp_utils import sample_init_points

MAXSAT_DIR_NAME = os.path.join(os.path.split(__file__)[0], 'maxsat2018_data')


class _MaxSAT(TestFunction):
	def __init__(self, data_filename, random_seed, normalize=False,  **kwargs):
		super(_MaxSAT, self).__init__(normalize, **kwargs)
		f = open(os.path.join(MAXSAT_DIR_NAME, data_filename), 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		self.n_variables = int(line_str.split(' ')[2])
		self.n_clauses = int(line_str.split(' ')[3])
		self.n_vertices = np.array([2] * self.n_variables)
		self.config = self.n_vertices
		clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
		f.close()
		weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
		weight_mean = np.mean(weights)
		weight_std = np.std(weights)
		self.weights = (weights - weight_mean) / weight_std
		self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses]

	def compute(self, x, normalize=None):
		if not isinstance(x, torch.Tensor):
			try:
				x = torch.tensor(x.astype(int))
			except:
				raise Exception('Unable to convert x to a pytorch tensor!')
		return self.evaluate(x)

	def evaluate(self, x,):

    #print(x.numel())
		assert x.numel() == self.n_variables
		if x.dim() == 2:
			x = x.squeeze(0)
		# 将 np.bool 改为 np.bool_
		x_np = (x.cpu() if x.is_cuda else x).numpy().astype(np.bool_)
		satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
		return -np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)


class MaxSAT28(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='maxcut-johnson8-2-4.clq.wcnf', random_seed=random_seed)


class MaxSAT43(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='maxcut-hamming8-2.clq.wcnf', random_seed=random_seed)


class MaxSAT60(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='frb-frb10-6-4.wcnf', random_seed=random_seed)


def maxsat60_function(x, dim=None, seed=None):
    """
    适配器函数，使MaxSAT60可以通过generate_discrete_test_dataset使用
    
    参数:
        x: 输入向量，元素为0或1
        dim: 输入维度，对于MaxSAT60应为60
        seed: 随机种子
    
    返回:
        float: 目标函数值
    """
    # 确保数据文件存在
    if not os.path.exists(os.path.join(MAXSAT_DIR_NAME, 'frb-frb10-6-4.wcnf')):
        download_maxsat60_data()
    
    # 创建MaxSAT60实例
    maxsat = MaxSAT60(random_seed=seed)
    
    # 检查维度是否正确
    if dim is not None and dim != maxsat.n_variables:
        raise ValueError(f"MaxSAT60的维度应为{maxsat.n_variables}，但提供了{dim}")
    
    # 转换输入格式
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(int))
    
    # 评估函数值
    return maxsat.evaluate(x)

# 注册MaxSAT60到test_functions字典
# def register_maxsat60_function():
#     """
#     将MaxSAT60注册到test_functions字典中，使其可以通过generate_discrete_test_dataset调用
#     """
#     # 导入test_functions字典
#     try:
#         from BOOM.test_funcs.synthetic_test_func import test_functions
        
#         # 添加maxsat60到test_functions
#         test_functions['maxsat60'] = {
#             'func': maxsat60_function,
#             'ranges': lambda d: np.array([[0, 1] for _ in range(d)]),
#             'dim_check': lambda d: d == 60,  # MaxSAT60的维度固定为60
#             'output_dim': 1
#         }
        
#         print("成功注册MaxSAT60到测试函数字典")
#         return True
#     except ImportError as e:
#         print(f"注册MaxSAT60失败: {e}")
#         return False

# 测试函数
if __name__ == '__main__':
    # 测试MaxSAT60类
    print("测试MaxSAT60类...")
    maxsat = MaxSAT60()
    x = torch.randint(0, 2, (maxsat.n_variables,))
    result = maxsat.evaluate(x)
    print(f"输入维度: {maxsat.n_variables}")
    print(f"函数值: {result}")
    
    # 测试适配器函数
    print("\n测试适配器函数...")
    x_np = np.random.randint(0, 2, maxsat.n_variables)
    result2 = maxsat60_function(x_np)
    print(f"适配器函数值: {result2}")
    
    # 注册MaxSAT60
    # print("\n注册MaxSAT60到测试函数字典...")
    # register_maxsat60_function()


# ------------------ MaxSAT Functions ------------------

def maxsat60_function(x, dim=None, seed=None):
    """
    适配器函数，使MaxSAT60可以通过generate_discrete_test_dataset使用

    参数:
        x: 输入向量，元素为0或1
        dim: 输入维度，对于MaxSAT60应为60
        seed: 随机种子

    返回:
        float: 目标函数值
    """
    # 确保数据文件存在

    # 创建MaxSAT60实例
    maxsat = MaxSAT60(random_seed=seed)

    # 检查维度是否正确
    if dim is not None and dim != maxsat.n_variables:
        raise ValueError(f"MaxSAT60的维度应为{maxsat.n_variables}，但提供了{dim}")

    # 转换输入格式
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(int))

     # 评估函数值
    result = maxsat.evaluate(x)

    # 确保返回的是标量而不是张量
    if isinstance(result, torch.Tensor):
        result = result.item()

    return float(result)

    # 评估函数值
    #return maxsat.evaluate(x)


import torch
from torch import Tensor
import numpy as np
import os
import requests
import tarfile
import shutil
from pathlib import Path
from typing import Optional, List

from test_functions.problem import DiscreteTestProblem

# MaxSAT数据目录路径
MAXSAT_DIR_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maxsat2018_data')
os.makedirs(MAXSAT_DIR_NAME, exist_ok=True)

class DiscreteMaxSAT60(DiscreteTestProblem):
    """
    Discrete MaxSAT60 problem.
    A Maximum Satisfiability problem with 60 variables.
    Each variable is a binary value (0 or 1).
    
    This implementation follows the style of DiscreteBranin and DiscretePestControl
    from the botorch test functions.
    """
    
    # 预定义类变量
    dim = 60
    num_objectives = 1
    
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.dim = 60  # MaxSAT60问题固定维度为60
        
        # 确保数据文件存在
        if not os.path.exists(os.path.join(MAXSAT_DIR_NAME, 'frb-frb10-6-4.wcnf')):
            download_maxsat60_data()
        
        # 设置_bounds - 所有变量为二元变量 (0或1)
        bounds = torch.zeros(self.dim, 2)
        bounds[:, 1] = 1  # 上界设为1
        
        self._bounds = bounds
        self.seed = seed

        self.maxsat = MaxSAT60(random_seed=seed)
        
        # 初始化内部的MaxSAT求解器
        self._init_maxsat_solver()
        
        # 调用父类初始化
        super().__init__(
            noise_std=noise_std,
            negate=negate,
            integer_indices=list(range(self.dim)),  # 所有维度都是整数值
            categorical_indices=None,
        )
    
    def _init_maxsat_solver(self):
        """初始化内部的MaxSAT求解器"""
        # 打开数据文件
        f = open(os.path.join(MAXSAT_DIR_NAME, 'frb-frb10-6-4.wcnf'), 'rt')
        line_str = f.readline()
        while line_str[:2] != 'p ':
            line_str = f.readline()
        self.n_variables = int(line_str.split(' ')[2])
        self.n_clauses = int(line_str.split(' ')[3])
        
        # 确保变量数量正确
        assert self.n_variables == self.dim, f"数据文件中的变量数量 ({self.n_variables}) 与预期的维度 ({self.dim}) 不匹配!"
        
        # 读取子句
        clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
        f.close()
        
        # 准备权重
        weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        self.weights = (weights - weight_mean) / weight_std
        
        # 准备子句
        self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses]
    
    # def evaluate_true(self, X: Tensor) -> Tensor:
    #     """
    #     评估MaxSAT60问题
        
    #     参数:
    #         X: shape (batch_size, dim) 的张量，其中每行代表一组0-1变量
        
    #     返回:
    #         shape (batch_size,) 的张量，表示每组变量的目标函数值
    #     """
    #     batch_size = X.shape[0]
    #     results = torch.zeros(batch_size, device=X.device, dtype=X.dtype)
        
    #     # 处理每个批次元素
    #     for i in range(batch_size):
    #         x_i = X[i]
    #         # 确保输入是二进制的
    #         x_i = x_i.round().to(torch.bool)
    #         # 计算满足的子句
    #         satisfied = torch.tensor([(x_i[torch.tensor(clause[0])] == torch.tensor(clause[1])).any() for clause in self.clauses], 
    #                                 device=X.device)
    #         # 计算目标函数值（加负号是因为我们想要最大化满足度，但DiscreteTestProblem假设是最小化问题）
    #         results[i] = torch.sum(torch.tensor(self.weights, device=X.device) * satisfied.float())
    #     print(X.shape)
    #     results = self.maxsat.evaluate(X)
        
    #     return results #.unsqueeze(-1)  # 添加额外的维度以匹配DiscreteTestProblem的预期输出形状

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        评估MaxSAT60问题
        
        参数:
            X: shape (batch_size, dim) 的张量，其中每行代表一组0-1变量
        
        返回:
            shape (batch_size, 1) 的张量，表示每组变量的目标函数值
        """
        # 确保X是二维张量
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        # 创建结果张量
        batch_size = X.shape[0]
        results = torch.zeros(batch_size, 1, device=X.device, dtype=X.dtype)
        
        # 处理每个批次元素
        for i in range(batch_size):
            x_i = X[i]
            # 使用maxsat.evaluate评估每个样本
            result = self.maxsat.evaluate(x_i)
            
            # 确保结果是标量，然后存储到结果张量中
            if result.dim() > 0:
                result = result.item() if result.numel() == 1 else result.squeeze()[0]
            
            results[i, 0] = result
        
        return results.squeeze(-1)  # 添加额外的维度以匹配DiscreteTestProblem的预期输出形状

