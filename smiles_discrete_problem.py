import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import qmc
import os

class SMILESDiscreteProblem:
    """
    将SMILES分子数据转换为离散优化问题形式，用于非LLM的优化算法。
    此类将SMILES转换为固定长度的离散特征向量，并维护原始SMILES与特征向量的映射。
    """
    
    def __init__(
        self,
        smiles_data_path: str,
        property_name: str,
        feature_type: str = 'morgan',
        fp_radius: int = 2,
        fp_bits: int = 60,
        discretize: bool = True,
        n_bins: int = 2,
        maximize: bool = True,
        seed: int = 42
    ):
        """
        初始化SMILES离散问题转换器
        
        参数:
        ----
        smiles_data_path: SMILES数据的CSV文件路径
        property_name: 要优化的属性名称
        feature_type: 特征类型 ('morgan', 'maccs', 'rdkit')
        fp_radius: Morgan指纹的半径
        fp_bits: 特征向量的长度/维度
        discretize: 是否将连续特征离散化
        n_bins: 离散化的箱数 (对每个特征)
        maximize: 是否最大化目标属性
        seed: 随机种子
        """
        self.smiles_data_path = smiles_data_path
        self.property_name = property_name
        self.feature_type = feature_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.discretize = discretize
        self.n_bins = n_bins
        self.maximize = maximize
        self.seed = seed
        self.num_objectives = 1  # 单目标优化
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据
        self._load_data()
        
        # 生成分子特征
        self._generate_features()
        
        # 如果需要离散化，则进行离散化处理
        if self.discretize:
            self._discretize_features()
        
        # 准备优化问题配置
        self._prepare_problem()
    
    def _load_data(self):
        """加载SMILES数据"""
        print(f"加载数据: {self.smiles_data_path}")
        self.df = pd.read_csv(self.smiles_data_path)
        
        # 检查必要的列是否存在
        if 'SMILES' not in self.df.columns:
            raise ValueError("数据集必须包含'SMILES'列")
        if self.property_name not in self.df.columns:
            raise ValueError(f"数据集必须包含'{self.property_name}'列")
        
        # 移除无效的SMILES
        valid_smiles = []
        valid_indices = []
        for i, smiles in enumerate(self.df['SMILES']):
            if pd.isna(smiles) or not isinstance(smiles, str):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
                valid_indices.append(i)
        
        # 过滤数据集只保留有效的SMILES
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"共加载 {len(self.df)} 个有效分子")
    
    def _generate_features(self):
        """为每个分子生成特征向量"""
        print(f"生成 {self.feature_type} 特征...")
        features = []
        valid_indices = []
        
        for i, smiles in enumerate(self.df['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 根据指定的特征类型计算特征
            if self.feature_type.lower() == 'morgan' :
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            elif self.feature_type.lower() == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif self.feature_type.lower() == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=self.fp_bits)
            else:
                raise ValueError(f"不支持的特征类型: {self.feature_type}")
            
            # 将位向量转换为numpy数组
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            # 如果使用MACCS密钥而特征长度小于指定的fp_bits，则填充到正确长度
            if len(arr) < self.fp_bits:
                arr = np.pad(arr, (0, self.fp_bits - len(arr)), 'constant')
            # 如果特征长度大于指定的fp_bits，则截断
            elif len(arr) > self.fp_bits:
                arr = arr[:self.fp_bits]
            
            features.append(arr)
            valid_indices.append(i)
        
        # 创建特征矩阵
        self.X = np.vstack(features)
        self.valid_indices = valid_indices
        self.valid_df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # 创建SMILES到特征和特征到SMILES的映射
        self.smiles_to_feature = {smiles: idx for idx, smiles in enumerate(self.valid_df['SMILES'])}
        self.feature_to_smiles = {idx: smiles for idx, smiles in enumerate(self.valid_df['SMILES'])}
        
        # 目标值
        self.y = self.valid_df[self.property_name].values
        # 如果是最小化问题而不是最大化问题，取反
        # if not self.maximize:
        #     self.y = -self.y
        
        print(f"特征矩阵形状: {self.X.shape}")
    
    def _discretize_features(self):
        """将连续特征离散化为指定数量的箱子"""
        print(f"将特征离散化为 {self.n_bins} 个箱...")
        
        # 创建离散化特征矩阵
        self.X_discrete = np.zeros_like(self.X, dtype=int)
        
        # 对每个特征进行离散化
        for i in range(self.X.shape[1]):
            feature_values = self.X[:, i]
            
            # 如果特征已经是二进制的 (0或1)，保持不变
            if set(np.unique(feature_values)) <= {0, 1}:
                self.X_discrete[:, i] = feature_values
                continue
            
            # 否则，创建等间隔的箱
            # 确保不会有NaN或无穷大的值
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
        if self.discretize:
            # 离散特征的范围是箱的索引
            self.bounds = np.zeros((self.dim, 2), dtype=int)
            for i in range(self.dim):
                if hasattr(self, 'X_discrete'):
                    unique_values = np.unique(self.X_discrete[:, i])
                    if len(unique_values) == 0:
                        # 如果没有唯一值，设置默认范围
                        self.bounds[i] = [0, 1]
                    else:
                        self.bounds[i] = [unique_values.min(), unique_values.max()]
                else:
                    # 默认为二值范围
                    self.bounds[i] = [0, 1]
        else:
            # 连续特征的范围是实际的最小值和最大值
            self.bounds = np.zeros((self.dim, 2))
            for i in range(self.dim):
                min_val = np.min(self.X[:, i])
                max_val = np.max(self.X[:, i])
                
                # 确保最小值和最大值不同，防止归一化出现NaN
                if min_val == max_val:
                    # 如果所有值都相同，设置一个小的范围
                    if min_val == 0:
                        self.bounds[i] = [0, 1]
                    else:
                        self.bounds[i] = [min_val * 0.9, min_val * 1.1]
                else:
                    self.bounds[i] = [min_val, max_val]
        
        # 转换为PyTorch张量以便与BoTorch兼容
        self.bounds_tensor = torch.tensor(self.bounds, dtype=torch.float64)
        # 转置边界张量为形状[2, dim]，符合BoTorch期望的格式
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
    
    def evaluate_true(self, x):
        """BoTorch测试函数接口兼容方法"""
        return self.evaluate(x) if not self.maximize else self.evaluate(x)
    
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
            distances = np.sum((self.X_discrete - x) ** 2, axis=1)
        else:
            distances = np.sum((self.X - x) ** 2, axis=1)
        
        return np.argmin(distances)
    
    def get_smiles_from_feature(self, x):
        """
        从特征向量获取对应的SMILES
        
        参数:
        ----
        x: 特征向量
        
        返回:
        ----
        对应的SMILES字符串
        """
        # 确保输入是numpy数组
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # 找到最接近的特征索引
        idx = self._find_nearest_feature(x_np)
        # 返回对应的SMILES
        return self.feature_to_smiles[idx]
    
    def sample_sobol_points(self, n_points):
        """
        使用Sobol序列采样特征空间
        
        参数:
        ----
        n_points: 要采样的点数
        
        返回:
        ----
        采样的特征向量和对应的目标值
        """
        print(f"使用Sobol序列采样 {n_points} 个点...")
        
        # 创建Sobol生成器
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        sobol_points = sobol.random(n=n_points)
        
        # 将[0,1]范围的点映射到每个特征的范围
        sampled_x = np.zeros((n_points, self.dim))
        for i in range(self.dim):
            # 映射到特征范围
            low, high = self.bounds[i]
            sampled_x[:, i] = sobol_points[:, i] * (high - low) + low
        
        # 如果是离散特征，取整
        if self.discretize:
            sampled_x = np.round(sampled_x).astype(int)
        
        # 评估样本点
        sampled_y = self.evaluate(sampled_x)
        
        # 转换为张量
        X_tensor = torch.tensor(sampled_x, dtype=torch.float64)
        Y_tensor = torch.tensor(sampled_y, dtype=torch.float64).unsqueeze(-1)
        
        return X_tensor, Y_tensor
    
    def sample_smiles_points(self, n_points):
        """
        随机采样原始SMILES数据集
        
        参数:
        ----
        n_points: 要采样的点数
        
        返回:
        ----
        采样的SMILES字符串和对应的目标值
        """
        if n_points > len(self.valid_df):
            print(f"警告: 请求的采样点数 ({n_points}) 大于可用的分子数 ({len(self.valid_df)})")
            n_points = len(self.valid_df)
        
        # 随机采样索引
        indices = np.random.choice(len(self.valid_df), size=n_points, replace=False)
        
        # 获取对应的SMILES和目标值
        sampled_smiles = [self.valid_df.iloc[i]['SMILES'] for i in indices]
        sampled_values = [self.valid_df.iloc[i][self.property_name] for i in indices]
        
        return sampled_smiles, sampled_values
    
    def save(self, path):
        """
        保存问题设置和转换器
        
        参数:
        ----
        path: 保存路径
        """
        import pickle
        
        save_dict = {
            'feature_type': self.feature_type,
            'fp_radius': self.fp_radius,
            'fp_bits': self.fp_bits,
            'discretize': self.discretize,
            'n_bins': self.n_bins,
            'maximize': self.maximize,
            'X': self.X,
            'y': self.y,
            'bounds': self.bounds,
            'valid_df': self.valid_df,
            'smiles_to_feature': self.smiles_to_feature,
            'feature_to_smiles': self.feature_to_smiles
        }
        
        if self.discretize and hasattr(self, 'X_discrete'):
            save_dict['X_discrete'] = self.X_discrete
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"已保存到 {path}")
    
    @classmethod
    def load(cls, path):
        """
        从保存的文件加载问题设置和转换器
        
        参数:
        ----
        path: 加载路径
        
        返回:
        ----
        SMILESDiscreteProblem实例
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
        instance.bounds_tensor = torch.tensor(instance.bounds, dtype=torch.float64)
        
        # 确保边界维度正确
        instance._bounds = torch.stack([
            torch.tensor([b[0] for b in instance.bounds], dtype=torch.float64),
            torch.tensor([b[1] for b in instance.bounds], dtype=torch.float64)
        ])
        
        instance.num_objectives = 1
        
        print(f"已从 {path} 加载")
        return instance


def create_debug_normalizer(problem):
    """
    为问题创建一个调试用的标准化函数，可以检查标准化过程中是否产生NaN
    
    参数:
    ----
    problem: SMILESDiscreteProblem实例
    
    返回:
    ----
    normalize_fn: 标准化函数
    unnormalize_fn: 反标准化函数
    """
    def debug_normalize(x):
        """将x标准化到[0,1]区间，并检查NaN"""
        bounds = problem._bounds.to(x.device)
        normalized = (x - bounds[0]) / (bounds[1] - bounds[0])
        
        # 检查是否有NaN
        if torch.isnan(normalized).any():
            print("警告: 标准化产生了NaN!")
            print("输入张量:", x)
            print("边界:", bounds)
            print("差值:", bounds[1] - bounds[0])
            print("标准化结果:", normalized)
            
            # 找出NaN的位置
            nan_indices = torch.where(torch.isnan(normalized))
            print("NaN位置:", nan_indices)
            
            # 打印对应位置的值和边界
            for i in range(len(nan_indices[0])):
                batch_idx = nan_indices[0][i].item()
                feat_idx = nan_indices[1][i].item()
                print(f"NaN位置[{batch_idx}, {feat_idx}]: x={x[batch_idx, feat_idx]}, "
                      f"bounds=[{bounds[0, feat_idx]}, {bounds[1, feat_idx]}], "
                      f"差值={bounds[1, feat_idx] - bounds[0, feat_idx]}")
            
            # 修复NaN：如果是由于边界相同导致的，则将该位置的值设为0.5
            for i in range(len(nan_indices[0])):
                batch_idx = nan_indices[0][i].item()
                feat_idx = nan_indices[1][i].item()
                if bounds[0, feat_idx] == bounds[1, feat_idx]:
                    normalized[batch_idx, feat_idx] = 0.5
            
            print("修复后的标准化结果:", normalized)
        
        return normalized
    
    def debug_unnormalize(normalized_x):
        """将[0,1]区间的x反标准化回原始区间"""
        bounds = problem._bounds.to(normalized_x.device)
        return bounds[0] + normalized_x * (bounds[1] - bounds[0])
    
    return debug_normalize, debug_unnormalize


# 用法示例
if __name__ == "__main__":
    # 示例CSV数据路径
    smiles_data_path = "data/redox_mer.csv"
    
    # 创建问题实例
    problem = SMILESDiscreteProblem(
        smiles_data_path=smiles_data_path,
        property_name="Ered",
        feature_type='morgan',
        fp_bits=60,  # 选择60维以匹配MaxSAT60
        discretize=True,
        n_bins=2,     # 二元离散化
        maximize=False  # 最小化还原电位
    )
    
    # 创建调试用的标准化函数
    debug_normalize, debug_unnormalize = create_debug_normalizer(problem)
    
    # 生成一些示例特征向量
    X_sample, Y_sample = problem.sample_sobol_points(10)
    
    # 尝试标准化
    normalized_X = debug_normalize(X_sample)
    print("标准化后的特征向量:")
    print(normalized_X[:3])
    
    # 测试反标准化
    unnormalized_X = debug_unnormalize(normalized_X)
    print("反标准化后的特征向量:")
    print(unnormalized_X[:3])
    
    # 检查与原始特征向量的差异
    print("原始与反标准化后的差异:")
    print(torch.abs(X_sample - unnormalized_X).max().item())
    
    # 获取对应的SMILES
    for i in range(3):
        x = X_sample[i]
        y = Y_sample[i]
        smiles = problem.get_smiles_from_feature(x)
        print(f"特征向量 {i+1} 对应的SMILES: {smiles}, 目标值: {y.item()}")


import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import qmc
import os

class SMILESDiscreteProblemMulti:
    """
    将SMILES分子数据转换为多目标离散优化问题形式，用于非LLM的优化算法。
    此类将SMILES转换为固定长度的离散特征向量，并维护原始SMILES与特征向量的映射。
    支持多个目标属性的同时优化。
    """
    
    def __init__(
        self,
        smiles_data_path: str,
        objective_names: list,
        feature_type: str = 'morgan',
        fp_radius: int = 2,
        fp_bits: int = 60,
        discretize: bool = True,
        n_bins: int = 2,
        maximize: list = None,
        seed: int = 42
    ):
        """
        初始化多目标SMILES离散问题转换器
        
        参数:
        ----
        smiles_data_path: SMILES数据的CSV文件路径
        objective_names: 要优化的多个属性名称列表
        feature_type: 特征类型 ('morgan', 'maccs', 'rdkit')
        fp_radius: Morgan指纹的半径
        fp_bits: 特征向量的长度/维度
        discretize: 是否将连续特征离散化
        n_bins: 离散化的箱数 (对每个特征)
        maximize: 对每个目标是否最大化的布尔值列表，默认全部最大化
        seed: 随机种子
        """
        self.smiles_data_path = smiles_data_path
        self.objective_names = objective_names
        self.feature_type = feature_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.discretize = discretize
        self.n_bins = n_bins
        self.maximize = maximize if maximize is not None else [True] * len(objective_names)
        self.seed = seed
        self.num_objectives = len(objective_names)
        
        # 验证输入
        if len(self.maximize) != self.num_objectives:
            raise ValueError(f"maximize列表的长度({len(self.maximize)})必须等于目标数量({self.num_objectives})")
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载数据
        self._load_data()
        
        # 生成分子特征
        self._generate_features()
        
        # 如果需要离散化，则进行离散化处理
        if self.discretize:
            self._discretize_features()
        
        # 准备优化问题配置
        self._prepare_problem()
    
    def _load_data(self):
        """加载SMILES数据"""
        print(f"加载数据: {self.smiles_data_path}")
        self.df = pd.read_csv(self.smiles_data_path)
        
        # 检查必要的列是否存在
        if 'SMILES' not in self.df.columns:
            raise ValueError("数据集必须包含'SMILES'列")
        
        for obj_name in self.objective_names:
            if obj_name not in self.df.columns:
                raise ValueError(f"数据集必须包含'{obj_name}'列")
        
        # 移除无效的SMILES和缺失的目标值
        valid_smiles = []
        valid_indices = []
        for i, row in self.df.iterrows():
            smiles = row['SMILES']
            
            # 检查SMILES是否有效
            if pd.isna(smiles) or not isinstance(smiles, str):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 检查所有目标值是否都存在且不为NaN
            if any(pd.isna(row[obj_name]) for obj_name in self.objective_names):
                continue
            
            valid_smiles.append(smiles)
            valid_indices.append(i)
        
        # 过滤数据集只保留有效的SMILES和完整的目标值
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"共加载 {len(self.df)} 个有效分子")
        print(f"目标属性: {self.objective_names}")
        print(f"目标统计:")
        for obj_name in self.objective_names:
            values = self.df[obj_name]
            print(f"  {obj_name}: 平均={values.mean():.4f}, 标准差={values.std():.4f}, 范围=[{values.min():.4f}, {values.max():.4f}]")
    
    def _generate_features(self):
        """为每个分子生成特征向量"""
        print(f"生成 {self.feature_type} 特征...")
        features = []
        valid_indices = []
        
        for i, smiles in enumerate(self.df['SMILES']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 根据指定的特征类型计算特征
            if self.feature_type.lower() == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
            elif self.feature_type.lower() == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            elif self.feature_type.lower() == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=self.fp_bits)
            else:
                raise ValueError(f"不支持的特征类型: {self.feature_type}")
            
            # 将位向量转换为numpy数组
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            # 如果使用MACCS密钥而特征长度小于指定的fp_bits，则填充到正确长度
            if len(arr) < self.fp_bits:
                arr = np.pad(arr, (0, self.fp_bits - len(arr)), 'constant')
            # 如果特征长度大于指定的fp_bits，则截断
            elif len(arr) > self.fp_bits:
                arr = arr[:self.fp_bits]
            
            features.append(arr)
            valid_indices.append(i)
        
        # 创建特征矩阵
        self.X = np.vstack(features)
        self.valid_indices = valid_indices
        self.valid_df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # 创建SMILES到特征和特征到SMILES的映射
        self.smiles_to_feature = {smiles: idx for idx, smiles in enumerate(self.valid_df['SMILES'])}
        self.feature_to_smiles = {idx: smiles for idx, smiles in enumerate(self.valid_df['SMILES'])}
        
        # 多目标值矩阵
        self.y = np.zeros((len(self.valid_df), self.num_objectives))
        for i, obj_name in enumerate(self.objective_names):
            self.y[:, i] = self.valid_df[obj_name].values
        
        print(f"特征矩阵形状: {self.X.shape}")
        print(f"目标矩阵形状: {self.y.shape}")
    
    def _discretize_features(self):
        """将连续特征离散化为指定数量的箱子"""
        print(f"将特征离散化为 {self.n_bins} 个箱...")
        
        # 创建离散化特征矩阵
        self.X_discrete = np.zeros_like(self.X, dtype=int)
        
        # 对每个特征进行离散化
        for i in range(self.X.shape[1]):
            feature_values = self.X[:, i]
            
            # 如果特征已经是二进制的 (0或1)，保持不变
            if set(np.unique(feature_values)) <= {0, 1}:
                self.X_discrete[:, i] = feature_values
                continue
            
            # 否则，创建等间隔的箱
            # 确保不会有NaN或无穷大的值
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
        if self.discretize:
            # 离散特征的范围是箱的索引
            self.bounds = np.zeros((self.dim, 2), dtype=int)
            for i in range(self.dim):
                if hasattr(self, 'X_discrete'):
                    unique_values = np.unique(self.X_discrete[:, i])
                    if len(unique_values) == 0:
                        # 如果没有唯一值，设置默认范围
                        self.bounds[i] = [0, 1]
                    else:
                        self.bounds[i] = [unique_values.min(), unique_values.max()]
                else:
                    # 默认为二值范围
                    self.bounds[i] = [0, 1]
        else:
            # 连续特征的范围是实际的最小值和最大值
            self.bounds = np.zeros((self.dim, 2))
            for i in range(self.dim):
                min_val = np.min(self.X[:, i])
                max_val = np.max(self.X[:, i])
                
                # 确保最小值和最大值不同，防止归一化出现NaN
                if min_val == max_val:
                    # 如果所有值都相同，设置一个小的范围
                    if min_val == 0:
                        self.bounds[i] = [0, 1]
                    else:
                        self.bounds[i] = [min_val * 0.9, min_val * 1.1]
                else:
                    self.bounds[i] = [min_val, max_val]
        
        # 转换为PyTorch张量以便与BoTorch兼容
        self.bounds_tensor = torch.tensor(self.bounds, dtype=torch.float64)
        # 转置边界张量为形状[2, dim]，符合BoTorch期望的格式
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
        评估给定特征向量对应的多目标属性值
        
        参数:
        ----
        x: 特征向量或向量的批次
        
        返回:
        ----
        多目标属性值矩阵，形状为 [n_points, n_objectives] 或 [n_objectives] (单点)
        """
        # 确保输入是numpy数组或PyTorch张量
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            is_tensor = True
            device = x.device
            dtype = x.dtype
        else:
            x_np = np.array(x)
            is_tensor = False
        
        # 如果是单个样本，添加批次维度
        single_point = False
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            single_point = True
        
        # 初始化结果
        results = np.zeros((len(x_np), self.num_objectives))
        
        for i, xi in enumerate(x_np):
            # 找到与给定特征向量最匹配的数据点
            best_match_idx = self._find_nearest_feature(xi)
            results[i] = self.y[best_match_idx]
        
        # 如果输入是单点，返回单点结果
        if single_point:
            results = results[0]
        
        # 转换为张量，如果输入是张量
        if is_tensor:
            results_tensor = torch.tensor(results, device=device, dtype=dtype)
            return results_tensor
        
        return results
    
    def evaluate_true(self, x):
        """BoTorch测试函数接口兼容方法"""
        return self.evaluate(x)
    
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
            distances = np.sum((self.X_discrete - x) ** 2, axis=1)
        else:
            distances = np.sum((self.X - x) ** 2, axis=1)
        
        return np.argmin(distances)
    
    def get_smiles_from_feature(self, x):
        """
        从特征向量获取对应的SMILES
        
        参数:
        ----
        x: 特征向量
        
        返回:
        ----
        对应的SMILES字符串
        """
        # 确保输入是numpy数组
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # 找到最接近的特征索引
        idx = self._find_nearest_feature(x_np)
        # 返回对应的SMILES
        return self.feature_to_smiles[idx]
    
    def sample_sobol_points(self, n_points):
        """
        使用Sobol序列采样特征空间
        
        参数:
        ----
        n_points: 要采样的点数
        
        返回:
        ----
        采样的特征向量和对应的多目标值
        """
        print(f"使用Sobol序列采样 {n_points} 个点...")
        
        # 创建Sobol生成器
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=self.seed)
        sobol_points = sobol.random(n=n_points)
        
        # 将[0,1]范围的点映射到每个特征的范围
        sampled_x = np.zeros((n_points, self.dim))
        for i in range(self.dim):
            # 映射到特征范围
            low, high = self.bounds[i]
            sampled_x[:, i] = sobol_points[:, i] * (high - low) + low
        
        # 如果是离散特征，取整
        if self.discretize:
            sampled_x = np.round(sampled_x).astype(int)
        
        # 评估样本点
        sampled_y = self.evaluate(sampled_x)
        
        # 转换为张量
        X_tensor = torch.tensor(sampled_x, dtype=torch.float64)
        Y_tensor = torch.tensor(sampled_y, dtype=torch.float64)
        
        # 确保Y_tensor有正确的形状
        if Y_tensor.dim() == 1:
            Y_tensor = Y_tensor.unsqueeze(-1)
        
        return X_tensor, Y_tensor
    
    def sample_smiles_points(self, n_points):
        """
        随机采样原始SMILES数据集
        
        参数:
        ----
        n_points: 要采样的点数
        
        返回:
        ----
        采样的SMILES字符串和对应的多目标值
        """
        if n_points > len(self.valid_df):
            print(f"警告: 请求的采样点数 ({n_points}) 大于可用的分子数 ({len(self.valid_df)})")
            n_points = len(self.valid_df)
        
        # 随机采样索引
        indices = np.random.choice(len(self.valid_df), size=n_points, replace=False)
        
        # 获取对应的SMILES和目标值
        sampled_smiles = [self.valid_df.iloc[i]['SMILES'] for i in indices]
        sampled_values = []
        for i in indices:
            row_values = [self.valid_df.iloc[i][obj_name] for obj_name in self.objective_names]
            sampled_values.append(row_values)
        
        return sampled_smiles, sampled_values
    
    def get_pareto_front_from_data(self):
        """
        从原始数据中计算帕累托前沿
        
        返回:
        ----
        pareto_smiles: 帕累托前沿上的SMILES列表
        pareto_values: 帕累托前沿上的目标值矩阵
        pareto_indices: 帕累托前沿点在原数据中的索引
        """
        from botorch.utils.multi_objective.pareto import is_non_dominated
        
        # 将目标值转换为张量
        Y = torch.tensor(self.y, dtype=torch.float64)
        
        # 根据最大化/最小化设置调整目标值
        adjusted_Y = Y.clone()
        for i, maximize in enumerate(self.maximize):
            if not maximize:
                adjusted_Y[:, i] = -adjusted_Y[:, i]
        
        # 计算帕累托前沿
        pareto_mask = is_non_dominated(adjusted_Y)
        pareto_indices = torch.where(pareto_mask)[0].numpy()
        
        # 获取帕累托前沿的SMILES和值
        pareto_smiles = [self.valid_df.iloc[i]['SMILES'] for i in pareto_indices]
        pareto_values = self.y[pareto_indices]
        
        return pareto_smiles, pareto_values, pareto_indices
    
    def save(self, path):
        """
        保存问题设置和转换器
        
        参数:
        ----
        path: 保存路径
        """
        import pickle
        
        save_dict = {
            'objective_names': self.objective_names,
            'feature_type': self.feature_type,
            'fp_radius': self.fp_radius,
            'fp_bits': self.fp_bits,
            'discretize': self.discretize,
            'n_bins': self.n_bins,
            'maximize': self.maximize,
            'num_objectives': self.num_objectives,
            'X': self.X,
            'y': self.y,
            'bounds': self.bounds,
            'valid_df': self.valid_df,
            'smiles_to_feature': self.smiles_to_feature,
            'feature_to_smiles': self.feature_to_smiles
        }
        
        if self.discretize and hasattr(self, 'X_discrete'):
            save_dict['X_discrete'] = self.X_discrete
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"已保存到 {path}")
    
    @classmethod
    def load(cls, path):
        """
        从保存的文件加载问题设置和转换器
        
        参数:
        ----
        path: 加载路径
        
        返回:
        ----
        SMILESDiscreteProblemMulti实例
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
        instance.bounds_tensor = torch.tensor(instance.bounds, dtype=torch.float64)
        
        # 确保边界维度正确
        instance._bounds = torch.stack([
            torch.tensor([b[0] for b in instance.bounds], dtype=torch.float64),
            torch.tensor([b[1] for b in instance.bounds], dtype=torch.float64)
        ])
        
        print(f"已从 {path} 加载")
        return instance


def create_debug_normalizer_multi(problem):
    """
    为多目标问题创建一个调试用的标准化函数，可以检查标准化过程中是否产生NaN
    
    参数:
    ----
    problem: SMILESDiscreteProblemMulti实例
    
    返回:
    ----
    normalize_fn: 标准化函数
    unnormalize_fn: 反标准化函数
    """
    def debug_normalize(x):
        """将x标准化到[0,1]区间，并检查NaN"""
        bounds = problem._bounds.to(x.device)
        normalized = (x - bounds[0]) / (bounds[1] - bounds[0])
        
        # 检查是否有NaN
        if torch.isnan(normalized).any():
            print("警告: 标准化产生了NaN!")
            print("输入张量:", x)
            print("边界:", bounds)
            print("差值:", bounds[1] - bounds[0])
            print("标准化结果:", normalized)
            
            # 找出NaN的位置
            nan_indices = torch.where(torch.isnan(normalized))
            print("NaN位置:", nan_indices)
            
            # 修复NaN：如果是由于边界相同导致的，则将该位置的值设为0.5
            for i in range(len(nan_indices[0])):
                batch_idx = nan_indices[0][i].item()
                feat_idx = nan_indices[1][i].item()
                if bounds[0, feat_idx] == bounds[1, feat_idx]:
                    normalized[batch_idx, feat_idx] = 0.5
            
            print("修复后的标准化结果:", normalized)
        
        return normalized
    
    def debug_unnormalize(normalized_x):
        """将[0,1]区间的x反标准化回原始区间"""
        bounds = problem._bounds.to(normalized_x.device)
        return bounds[0] + normalized_x * (bounds[1] - bounds[0])
    
    return debug_normalize, debug_unnormalize


# 用法示例
if __name__ == "__main__":
    # 示例：假设你有一个包含多个目标的CSV文件
    # CSV文件应该包含: SMILES, objective1, objective2, objective3 等列
    
    # 示例配置
    smiles_data_path = "data/multi_objective_data.csv"  # 替换为你的文件路径
    objective_names = ["objective1", "objective2", "objective3"]  # 替换为你的目标名称
    
    # 创建多目标问题实例
    problem = SMILESDiscreteProblemMulti(
        smiles_data_path=smiles_data_path,
        objective_names=objective_names,
        feature_type='morgan',
        fp_bits=60,
        discretize=True,
        n_bins=2,
        maximize=[True, False, True],  # 第一个和第三个目标最大化，第二个最小化
        seed=42
    )
    
    # 创建调试用的标准化函数
    debug_normalize, debug_unnormalize = create_debug_normalizer_multi(problem)
    
    # 生成一些示例特征向量
    X_sample, Y_sample = problem.sample_sobol_points(10)
    
    print("多目标优化问题配置:")
    print(f"目标数量: {problem.num_objectives}")
    print(f"目标名称: {problem.objective_names}")
    print(f"最大化设置: {problem.maximize}")
    print(f"特征向量形状: {X_sample.shape}")
    print(f"目标值形状: {Y_sample.shape}")
    
    # 显示一些样本
    print("\n样本数据:")
    for i in range(3):
        x = X_sample[i]
        y = Y_sample[i]
        smiles = problem.get_smiles_from_feature(x)
        print(f"分子 {i+1}: {smiles}")
        for j, obj_name in enumerate(objective_names):
            print(f"  {obj_name}: {y[j].item():.4f}")
    
    # 获取数据集中的帕累托前沿
    pareto_smiles, pareto_values, pareto_indices = problem.get_pareto_front_from_data()
    print(f"\n数据集中的帕累托前沿大小: {len(pareto_smiles)}")
    print("帕累托前沿示例:")
    for i in range(min(3, len(pareto_smiles))):
        print(f"  {pareto_smiles[i]}: {pareto_values[i]}")