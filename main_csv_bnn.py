import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
import gzip
import shutil

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import normalize, unnormalize
from models import *  # 假设您的models.py保持不变
import wandb
import numpy as np
import pandas as pd
from scipy.stats import qmc

class DiscreteTableProblem:
    """
    处理表格数据的离散优化问题，适用于非SMILES的CSV表格数据。
    """
    
    def __init__(
        self,
        data_path: str,
        property_name: str,
        discretize: bool = False,
        n_bins: int = 2,
        maximize: bool = True,
        seed: int = 42
    ):
        """
        初始化表格数据优化问题
        
        参数:
        ----
        data_path: 表格数据的CSV文件路径
        property_name: 要优化的属性名称
        discretize: 是否将连续特征离散化
        n_bins: 离散化的箱数 (对每个特征)
        maximize: 是否最大化目标属性
        seed: 随机种子
        """
        self.data_path = data_path
        self.property_name = property_name
        self.discretize = discretize
        self.n_bins = n_bins
        self.maximize = maximize
        self.seed = seed
        self.num_objectives = 1  # 单目标优化
        
        # 设置随机种子
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        
        # 加载数据
        self._load_data()
        
        # 如果需要离散化，则进行离散化处理
        if self.discretize:
            self._discretize_features()
        
        # 准备优化问题配置
        self._prepare_problem()
    
    def _load_data(self):
        """加载表格数据"""
        print(f"加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # 检查必要的列是否存在
        if self.property_name not in self.df.columns:
            raise ValueError(f"数据集必须包含'{self.property_name}'列")
            
        # 提取特征列（假设特征列以x_开头）
        feature_cols = [col for col in self.df.columns if col.startswith('x_')]
        
        # 如果没有找到以x_开头的列，则使用所有非目标列作为特征
        if not feature_cols:
            feature_cols = [col for col in self.df.columns if col != self.property_name]
            print(f"未找到以x_开头的特征列，将使用所有非目标列作为特征: {feature_cols}")
        else:
            print(f"发现 {len(feature_cols)} 个特征列")
        
        # 创建特征矩阵和目标值
        self.X = self.df[feature_cols].values
        self.y = self.df[self.property_name].values
        
        # 行索引到特征的映射
        self.idx_to_row = {i: i for i in range(len(self.df))}
        
        print(f"数据形状: {self.X.shape}")
        print(f"目标列: {self.property_name}")
    
    def _discretize_features(self):
        """将连续特征离散化为指定数量的箱子"""
        print(f"将特征离散化为 {self.n_bins} 个箱...")
        
        # 创建离散化特征矩阵
        self.X_discrete = np.zeros_like(self.X, dtype=int)
        
        # 对每个特征进行离散化
        for i in range(self.X.shape[1]):
            feature_values = self.X[:, i]
            
            # 如果特征已经是离散的，保持不变
            unique_values = np.unique(feature_values)
            if len(unique_values) <= self.n_bins:
                self.X_discrete[:, i] = feature_values
                continue
            
            # 否则，创建等间隔的箱
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # 如果最小值等于最大值，则所有值都相同，直接赋值为0
            if min_val == max_val:
                self.X_discrete[:, i] = 0
                continue
                
            bins = np.linspace(min_val, max_val, self.n_bins + 1)
            
            # 执行离散化，得到箱索引 (0 到 n_bins-1)
            self.X_discrete[:, i] = np.digitize(feature_values, bins[1:-1])
        
        print(f"离散化后的特征矩阵形状: {self.X_discrete.shape}")
    
    def _prepare_problem(self):
        """准备优化问题的配置"""
        # 设置问题维度
        self.dim = self.X.shape[1]
        
        # 设置特征的取值范围
        self.bounds = np.zeros((self.dim, 2))
        if self.discretize and hasattr(self, 'X_discrete'):
            X_for_bounds = self.X_discrete
        else:
            X_for_bounds = self.X
            
        for i in range(self.dim):
            min_val = np.min(X_for_bounds[:, i])
            max_val = np.max(X_for_bounds[:, i])
            
            # 确保最小值和最大值不同，防止归一化出现NaN
            if min_val == max_val:
                # 如果所有值都相同，设置一个小的范围
                if min_val == 0:
                    self.bounds[i] = [0, 1]
                else:
                    self.bounds[i] = [min_val - 0.1, min_val + 0.1]
            else:
                self.bounds[i] = [min_val, max_val]
        
        # 转换为PyTorch张量以便与BoTorch兼容
        self._bounds = torch.stack([
            torch.tensor([b[0] for b in self.bounds], dtype=torch.float64),
            torch.tensor([b[1] for b in self.bounds], dtype=torch.float64)
        ])
        
        print(f"问题维度: {self.dim}")
        print(f"特征取值范围示例 (前5个特征): {self.bounds[:5]}")
    
    def __call__(self, x):
        """使类实例可调用，与标准测试函数格式一致"""
        return self.evaluate(x)
    
    def evaluate(self, x):
        """
        评估给定特征向量对应的目标属性值
        
        参数:
        ----
        x: 特征向量或向量的批次
        
        返回:
        ----
        目标属性值或值的批次
        """
        # 确保输入是numpy数组或PyTorch张量
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            is_tensor = True
        else:
            x_np = np.array(x)
            is_tensor = False
        
        # 如果是单个样本，添加批次维度
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        
        # 初始化结果
        results = np.zeros(len(x_np))
        
        for i, xi in enumerate(x_np):
            # 找到与给定特征向量最匹配的数据点
            best_match_idx = self._find_nearest_feature(xi)
            results[i] = self.y[best_match_idx]
        
        # 转换为张量，如果输入是张量
        if is_tensor:
            return torch.tensor(results, device=x.device, dtype=x.dtype)
        
        return results
    
    def _find_nearest_feature(self, x):
        """
        找到与给定特征向量最接近的数据点
        
        参数:
        ----
        x: 特征向量
        
        返回:
        ----
        最接近的数据点索引
        """
        # 使用欧几里得距离找到最接近的特征向量
        if self.discretize and hasattr(self, 'X_discrete'):
            # 将输入向量取整以匹配离散特征
            x_discrete = np.round(x).astype(int)
            distances = np.sum((self.X_discrete - x_discrete) ** 2, axis=1)
        else:
            distances = np.sum((self.X - x) ** 2, axis=1)
        
        return np.argmin(distances)
    
    def get_row_index(self, x):
        """
        从特征向量获取对应的行索引
        
        参数:
        ----
        x: 特征向量
        
        返回:
        ----
        对应的行索引
        """
        # 确保输入是numpy数组
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # 找到最接近的特征索引
        idx = self._find_nearest_feature(x_np)
        return idx
    
    def get_feature_values(self, idx):
        """获取指定行索引的特征值"""
        return self.X[idx]
    
    def sample_points(self, n_points):
        """
        从数据集中随机采样点
        
        参数:
        ----
        n_points: 要采样的点数
        
        返回:
        ----
        采样的特征向量和对应的目标值
        """
        print(f"从数据集中随机采样 {n_points} 个点...")
        
        if n_points > len(self.X):
            print(f"警告: 请求的采样点数 ({n_points}) 大于可用的数据点 ({len(self.X)})")
            n_points = len(self.X)
        
        # 随机采样索引
        indices = np.random.choice(len(self.X), size=n_points, replace=False)
        
        # 获取对应的特征和目标值
        sampled_x = self.X[indices]
        sampled_y = self.y[indices]
        
        # 转换为张量
        X_tensor = torch.tensor(sampled_x, dtype=torch.float64)
        Y_tensor = torch.tensor(sampled_y, dtype=torch.float64).unsqueeze(-1)
        
        return X_tensor, Y_tensor
    
    def save(self, path):
        """
        保存问题设置
        
        参数:
        ----
        path: 保存路径
        """
        import pickle
        
        save_dict = {
            'discretize': self.discretize,
            'n_bins': self.n_bins,
            'maximize': self.maximize,
            'X': self.X,
            'y': self.y,
            'bounds': self.bounds,
            'property_name': self.property_name
        }
        
        if self.discretize and hasattr(self, 'X_discrete'):
            save_dict['X_discrete'] = self.X_discrete
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"已保存到 {path}")
    
    @classmethod
    def load(cls, path):
        """
        从保存的文件加载问题设置
        
        参数:
        ----
        path: 加载路径
        
        返回:
        ----
        DiscreteTableProblem实例
        """
        import pickle
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # 创建实例但不初始化
        instance = cls.__new__(cls)
        
        # 设置属性
        for key, value in save_dict.items():
            setattr(instance, key, value)
        
        # 设置其他必要的属性
        instance.dim = instance.X.shape[1]
        instance.idx_to_row = {i: i for i in range(len(instance.X))}
        
        # 确保边界维度正确
        instance._bounds = torch.stack([
            torch.tensor([b[0] for b in instance.bounds], dtype=torch.float64),
            torch.tensor([b[1] for b in instance.bounds], dtype=torch.float64)
        ])
        
        instance.num_objectives = 1
        
        print(f"已从 {path} 加载")
        return instance


def bayes_opt_tabular(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
    """
    针对表格数据的贝叶斯优化函数
    
    参数:
    ----
    model: 模型实例
    problem: DiscreteTableProblem实例
    args: 配置参数
    init_x: 初始特征向量
    init_y: 初始目标值
    model_save_dir: 模型保存目录
    device: 计算设备
    model_name: 模型名称
    """
    q = int(args["batch_size"])
    output_dim = init_y.shape[-1]
    bounds = problem._bounds.to(init_x)

    standard_bounds = torch.zeros(2, problem.dim).to(init_x)
    standard_bounds[1] = 1

    train_x = init_x
    train_y = init_y

    print("初始点数量:", len(train_x))
    print("初始值:", train_y)

    # 记录对应的行索引
    train_indices = []
    for i in range(len(train_x)):
        idx = problem.get_row_index(train_x[i].cpu().numpy())
        train_indices.append(idx)
    
    # 获取完整的候选池
    all_features = torch.tensor(problem.X, dtype=torch.float64).to(device)
    all_targets = torch.tensor(problem.y, dtype=torch.float64).unsqueeze(-1).to(device)
    
    # 创建已评估点的索引集合
    evaluated_indices = set(train_indices)
    
    # 保存初始点集
    property_name = args["property_name"]
    with open(f"{model_save_dir}/initial_points.csv", "w") as f:
        f.write(f"Index,{property_name}\n")
        for idx, y in zip(train_indices, train_y):
            f.write(f"{idx},{y.item()}\n")

    # 使用wandb跟踪实验
    use_wandb = args.get("use_wandb", True)
    if use_wandb:
        wandb_project = args.get("wandb_project", "Tabular_Optimization")
        wandb.init(
            project=wandb_project,
            name=f"{model_name}-{args['property_name']}",
            config=args
        )

    # 主优化循环
    for i in range(args["n_BO_iters"]):
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

        # 在归一化的特征空间上拟合模型
        model_start = time.time()
        normalized_x = normalize(train_x, bounds).to(train_x)
        model.fit_and_save(normalized_x, train_y, model_save_dir)
        model_end = time.time()
        print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
        # 构建采集函数
        acq_start = time.time()
        sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
        # 使用期望改进作为采集函数
        best_f = train_y.max() if problem.maximize else -train_y.min()
        acquisition = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler
        )
        
        # 从候选池中选择未评估的点
        available_indices = [i for i in range(len(problem.X)) if i not in evaluated_indices]
        if not available_indices:
            print("所有点都已评估，优化提前结束")
            break
            
        available_x = all_features[available_indices]
        normalized_available_x = normalize(available_x, bounds)
        
        # 计算每个候选点的采集值
        with torch.no_grad():
            acq_values = acquisition(normalized_available_x.unsqueeze(1))
        
        # 选择采集值最高的点
        best_acq_idx = torch.argmax(acq_values)
        original_idx = available_indices[best_acq_idx]
        
        # 获取选中的点
        new_x = all_features[original_idx].to(train_x)
        
        # 获取真实值
        new_y = all_targets[original_idx].to(train_y)
        
        # 释放内存
        del acquisition
        del acq_values
        del normalized_available_x
        torch.cuda.empty_cache()
        
        acq_end = time.time()
        print(f"候选点选择时间: {acq_end - acq_start:.2f}秒")
        
        # 更新已评估点索引
        evaluated_indices.add(original_idx)
        train_indices.append(original_idx)
        
        # 扩展训练集
        train_x = torch.cat([train_x, new_x.unsqueeze(0)])
        train_y = torch.cat([train_y, new_y.unsqueeze(0)])
        
        # 打印当前最佳值
        if problem.maximize:
            best_idx = torch.argmax(train_y)
            best_val = train_y[best_idx].item()
            best_row_idx = train_indices[best_idx]
        else:
            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx].item()
            best_row_idx = train_indices[best_idx]
        
        # 获取最佳点的特征值
        best_features = problem.get_feature_values(best_row_idx)
        
        print(f"新评估的行索引: {original_idx}")
        print(f"新评估的值: {new_y.item() if output_dim == 1 else new_y}")
        print(f"当前最佳值: {best_val} (对应行索引: {best_row_idx})")
        print(f"最佳点特征: {best_features}")
        
        # 记录到wandb
        if use_wandb:
            wandb.log({
                "iteration": i + 1,
                "best_value": best_val,
                "actual_value": new_y.item() if output_dim == 1 else new_y[0].item(),
                "best_row_idx": best_row_idx
            })

    # 保存结果
    if model_save_dir is not None:
        torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
        torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
        # 保存所有评估过的点
        with open(f"{model_save_dir}/evaluated_points.csv", "w") as f:
            f.write(f"Row_Index,{property_name}")
            # 添加特征列
            for i in range(problem.dim):
                f.write(f",x_{i}")
            f.write("\n")
            
            for idx, y in zip(train_indices, train_y):
                features = problem.get_feature_values(idx)
                f.write(f"{idx},{y.item()}")
                for feat_val in features:
                    f.write(f",{feat_val}")
                f.write("\n")
        
        # 保存最佳点
        if problem.maximize:
            best_idx = torch.argmax(train_y)
            best_val = train_y[best_idx].item()
        else:
            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx].item()
        
        best_x = train_x[best_idx]
        best_row_idx = train_indices[best_idx]
        best_features = problem.get_feature_values(best_row_idx)
        
        with open(f"{model_save_dir}/best_point.txt", "w") as f:
            f.write(f"行索引: {best_row_idx}\n")
            f.write(f"{args['property_name']}: {best_val}\n")
            f.write("特征值:\n")
            for i, val in enumerate(best_features):
                f.write(f"x_{i}: {val}\n")

    if use_wandb:
        wandb.finish()

    return best_x, train_y[best_idx], best_row_idx


def initialize_model(model_name, model_args, input_dim, output_dim, device):
    """初始化模型"""
    if model_name == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    elif model_name == 'dkl':
        if output_dim == 1:
            return SingleTaskDKL(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskDKL(model_args, input_dim, output_dim, device)
    elif model_name == 'ibnn':
        if output_dim == 1:
            return SingleTaskIBNN(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskIBNN(model_args, input_dim, output_dim, device)
    elif model_name == 'hmc':
        return HMC(model_args, input_dim, output_dim, device)
    elif model_name == 'sghmc':
        return SGHMCModel(model_args, input_dim, output_dim, device)
    elif model_name == 'laplace':
        return LaplaceBNN(model_args, input_dim, output_dim, device)
    elif model_name == 'ensemble':
        return Ensemble(model_args, input_dim, output_dim, device)
    else:
        raise NotImplementedError(f"模型类型 {model_name} 不存在")


def main(cl_args):
    """主函数 - 用于表格数据优化"""
    current_time = datetime.now()
    args = json.load(open("./config/" + cl_args.config + ".json", 'r'))

    # 设置保存目录
    save_dir = current_time.strftime("experiment_results/%y_%m_%d-%H_%M_%S")
    property_name = args["property_name"]
    dataset_name = args["dataset_name"]
    
    if cl_args.name:
        save_dir = f"{save_dir}_{cl_args.name}_{dataset_name}_{property_name}"
    else:
        save_dir = f"{save_dir}_{cl_args.config}_{dataset_name}_{property_name}"
    
    os.makedirs(save_dir, exist_ok=True)

    try:
        if cl_args.bg:
            # 重定向输出
            sys.stdout = open(f"{save_dir}/stdout.txt", 'w')
            sys.stderr = open(f"{save_dir}/stderr.txt", 'w')

        # 保存配置
        with open(f"{save_dir}/config.json", 'w') as f:
            json.dump(args, f, indent=2)
        
        # 设置设备和随机种子
        device = torch.device("cpu")#('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_dtype(torch.float64)
        # torch.manual_seed(int(args["seed"]))
        # np.random.seed(int(args["seed"]))

        # 处理数据集路径
        # 检查dataset_name是否为完整路径
        if os.path.isabs(dataset_name) or dataset_name.startswith('./') or dataset_name.startswith('../'):
            # 如果是绝对路径或相对路径，直接使用
            dataset_path = dataset_name
        else:
            # 否则，拼接data_dir和dataset_name
            data_dir = args.get("data_dir", "./data")
            dataset_path = os.path.join(data_dir, dataset_name)
            
        print(f"使用数据集路径: {dataset_path}")
        
        # 检查是否需要解压
        if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
            print(f"解压数据文件: {dataset_path}.gz")
            with gzip.open(dataset_path + ".gz", 'rb') as f_in:
                with open(dataset_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        # 如果仍然找不到文件，检查是否需要添加.csv后缀
        if not os.path.exists(dataset_path) and not dataset_path.endswith('.csv'):
            dataset_path = dataset_path + ".csv"
            if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
                print(f"解压数据文件: {dataset_path}.gz")
                with gzip.open(dataset_path + ".gz", 'rb') as f_in:
                    with open(dataset_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到数据集文件: {dataset_path} 或 {dataset_path}.gz")

        # 创建优化问题
        print(f"创建表格数据优化问题...")
        problem = DiscreteTableProblem(
            data_path=dataset_path,
            property_name=property_name,
            discretize=args.get("discretize", False),
            n_bins=args.get("n_bins", 2),
            maximize=args.get("maximize", False),
            seed=int(args["seed"])
        )
        
        # 保存问题定义
        problem.save(os.path.join(save_dir, "problem.pkl"))
        
        # 获取维度
        input_dim = problem.dim
        output_dim = 1  # 单目标优化
        
        print(f"问题维度: {input_dim}")
        print(f"离散化: {problem.discretize}")
        print(f"目标: {'最大化' if problem.maximize else '最小化'} {property_name}")

        # 开始多次实验
        for trial in range(1, args["n_trials"] + 1):
            print("-" * 20, f"开始实验 {trial}" + "-" * 20)
            
            # 采样初始点
            n_init_points = args["n_init_points"]
            init_x, init_y = problem.sample_points(n_init_points)
            init_x = init_x.to(device)
            init_y = init_y.to(device)
            
            # 为每个模型运行贝叶斯优化
            model_dict = args["models"]
            for model_id, model_args in model_dict.items():
                model_name = model_args["model"]

                model_save_dir = f"{save_dir}/trial_{trial}/{model_id}"
                os.makedirs(model_save_dir, exist_ok=True)
                os.makedirs(f"{model_save_dir}/model_state", exist_ok=True)
                os.makedirs(f"{model_save_dir}/queries", exist_ok=True)

                print("-" * 20, f"运行 {model_id}" + "-" * 20)
                start_time = time.time()
                
                model = initialize_model(model_name, model_args, input_dim, output_dim, device)
                best_x, best_y, best_row_idx = bayes_opt_tabular(
                    model, problem, args, init_x, init_y, model_save_dir, device, model_name
                )
                del model

                print(f"\n找到的最佳值: {best_y.item()}")
                print(f"最佳行索引: {best_row_idx}")
                print(f"用时(秒): {time.time() - start_time:.2f}")

            torch.cuda.empty_cache()

        os.rename(save_dir, save_dir + "_done")
        print("优化完成!")
    
    except Exception as e:
        print(traceback.format_exc())
        os.rename(save_dir, save_dir + "_canceled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="表格数据优化程序")
    parser.add_argument("--config", type=str, default="tabular_default", help="配置文件名")
    parser.add_argument("--bg", action="store_true", help="是否在后台运行")
    parser.add_argument("-n", "--name", type=str, help="实验名称(可选)")
    cl_args = parser.parse_args()

    main(cl_args)