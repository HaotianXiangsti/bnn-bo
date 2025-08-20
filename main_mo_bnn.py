# import argparse
# import json
# import os
# import sys
# import time
# import traceback
# from datetime import datetime
# import gzip
# import shutil

# import torch
# from botorch.acquisition import qExpectedImprovement
# from botorch.optim import optimize_acqf
# from botorch.sampling.stochastic_samplers import StochasticSampler
# from botorch.utils.transforms import normalize, unnormalize
# from models import *
# import wandb
# import numpy as np

# from smiles_discrete_problem import SMILESDiscreteProblem

# def round_feature_vector(x, problem):
#     """根据问题类型对特征向量取整"""
#     if problem.discretize:
#         return torch.round(x)
#     return x

# def bayes_opt_molecular(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
#     """
#     针对分子优化的贝叶斯优化函数
    
#     参数:
#     ----
#     model: 模型实例
#     problem: SMILESDiscreteProblem实例
#     args: 配置参数
#     init_x: 初始特征向量
#     init_y: 初始目标值
#     model_save_dir: 模型保存目录
#     device: 计算设备
#     model_name: 模型名称
#     """
#     q = int(args["batch_size"])
#     output_dim = init_y.shape[-1]
#     bounds = problem._bounds.to(init_x)

#     standard_bounds = torch.zeros(2, problem.dim).to(init_x)
#     standard_bounds[1] = 1

#     train_x = init_x
#     train_y = init_y

#     print("初始点数量:", len(train_x))
#     print("初始值:", train_y)

#     # 记录对应的SMILES
#     train_smiles = []
#     for i in range(len(train_x)):
#         smiles = problem.get_smiles_from_feature(train_x[i].cpu().numpy())
#         train_smiles.append(smiles)
    
#     # 保存初始分子集
#     with open(f"{model_save_dir}/initial_molecules.csv", "w") as f:
#         f.write("SMILES," + args["property_name"] + "\n")
#         for smiles, y in zip(train_smiles, train_y):
#             f.write(f"{smiles},{y.item()}\n")

#     # 使用wandb跟踪实验
#     use_wandb = args.get("use_wandb", True)
#     if use_wandb:
#         wandb_project = args.get("wandb_project", "Molecular_Optimization")
#         wandb.init(
#             project=wandb_project,
#             name=f"{model_name}-{args['property_name']}",
#             config=args
#         )

#     # 主优化循环
#     for i in range(args["n_BO_iters"]):
#         sys.stdout.flush()
#         sys.stderr.flush()
#         print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

#         # 在归一化的特征空间上拟合模型
#         model_start = time.time()
#         normalized_x = train_x#normalize(train_x, bounds).to(train_x)
#         model.fit_and_save(normalized_x, train_y, model_save_dir)
#         model_end = time.time()
#         print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
#         # 构建采集函数
#         acq_start = time.time()
#         sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
#         # 使用期望改进作为采集函数
#         best_f = train_y.max() if problem.maximize else -train_y.min()
#         acquisition = qExpectedImprovement(
#             model=model,
#             best_f=best_f,
#             sampler=sampler
#         )
        
#         # 优化采集函数
#         normalized_candidates, acqf_values = optimize_acqf(
#             acquisition, standard_bounds, q=q, num_restarts=2, raw_samples=16, return_best_only=False,
#             options={"batch_limit": 1, "maxiter": 10})
#         candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)

#         # 对候选点取整
#         candidates = round_feature_vector(candidates, problem)
        
#         # 计算取整后的采集值
#         normalized_rounded_candidates = normalize(candidates, bounds)
#         acqf_values = acquisition(normalized_rounded_candidates)
#         acq_end = time.time()
#         print(f"采集函数优化时间: {acq_end - acq_start:.2f}秒")

#         # 选择最佳候选点
#         best_index = acqf_values.max(dim=0).indices.item()
#         new_x = candidates[best_index].to(train_x)

#         # 释放内存
#         del acquisition
#         del acqf_values
#         del normalized_candidates
#         del normalized_rounded_candidates
#         torch.cuda.empty_cache()

#         # 评估新点
#         new_y = problem(new_x)
#         # 添加输出维度
#         if output_dim == 1:
#             new_y = new_y.unsqueeze(-1)
        
#         # 获取对应的SMILES
#         new_smiles = problem.get_smiles_from_feature(new_x.cpu().numpy())
#         train_smiles.append(new_smiles)
        
#         # 扩展训练集
#         train_x = torch.cat([train_x, new_x])#.unsqueeze(0)])
#         train_y = torch.cat([train_y, new_y])
        
#         # 打印当前最佳值
#         if problem.maximize:
#             best_idx = torch.argmax(train_y)
#             best_val = train_y[best_idx].item()
#             best_smiles = train_smiles[best_idx]
#         else:
#             best_idx = torch.argmin(train_y)
#             best_val = train_y[best_idx].item()
#             best_smiles = train_smiles[best_idx]
        
#         print(f"新评估的分子: {new_smiles}")
#         print(f"新评估的值: {new_y.item() if output_dim == 1 else new_y}")
#         print(f"当前最佳值: {best_val} (对应分子: {best_smiles})")
        
#         # 记录到wandb
#         if use_wandb:
#             wandb.log({
#                 "iteration": i + 1,
#                 "best_value": best_val,
#                 "actual_value": new_y.item() if output_dim == 1 else new_y[0].item(),
#                 "best_molecule": best_smiles
#             })

#     # 保存结果
#     if model_save_dir is not None:
#         torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
#         torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
#         # 保存所有评估过的分子
#         with open(f"{model_save_dir}/evaluated_molecules.csv", "w") as f:
#             f.write("SMILES," + args["property_name"] + "\n")
#             for smiles, y in zip(train_smiles, train_y):
#                 f.write(f"{smiles},{y.item()}\n")
        
#         # 保存最佳分子
#         if problem.maximize:
#             best_idx = torch.argmax(train_y)
#             best_val = train_y[best_idx].item()
#         else:
#             best_idx = torch.argmin(train_y)
#             best_val = train_y[best_idx].item()
        
#         best_x = train_x[best_idx]
#         best_smiles = train_smiles[best_idx]
        
#         with open(f"{model_save_dir}/best_molecule.txt", "w") as f:
#             f.write(f"SMILES: {best_smiles}\n")
#             f.write(f"{args['property_name']}: {best_val}\n")

#     if use_wandb:
#         wandb.finish()

#     return best_x, train_y[best_idx], best_smiles


# def initialize_model(model_name, model_args, input_dim, output_dim, device):
#     """初始化模型"""
#     if model_name == 'gp':
#         if output_dim == 1:
#             return SingleTaskGP(model_args, input_dim, output_dim)
#         else:
#             return MultiTaskGP(model_args, input_dim, output_dim)
#     elif model_name == 'dkl':
#         if output_dim == 1:
#             return SingleTaskDKL(model_args, input_dim, output_dim, device)
#         else:
#             return MultiTaskDKL(model_args, input_dim, output_dim, device)
#     elif model_name == 'ibnn':
#         if output_dim == 1:
#             return SingleTaskIBNN(model_args, input_dim, output_dim, device)
#         else:
#             return MultiTaskIBNN(model_args, input_dim, output_dim, device)
#     elif model_name == 'hmc':
#         return HMC(model_args, input_dim, output_dim, device)
#     elif model_name == 'sghmc':
#         return SGHMCModel(model_args, input_dim, output_dim, device)
#     elif model_name == 'laplace':
#         return LaplaceBNN(model_args, input_dim, output_dim, device)
#     elif model_name == 'ensemble':
#         return Ensemble(model_args, input_dim, output_dim, device)
#     else:
#         raise NotImplementedError(f"模型类型 {model_name} 不存在")


# def main(cl_args):
#     """主函数"""
#     current_time = datetime.now()
#     args = json.load(open("./config/" + cl_args.config + ".json", 'r'))

#     # 设置保存目录
#     save_dir = current_time.strftime("experiment_results/%y_%m_%d-%H_%M_%S")
#     property_name = args["property_name"]
#     dataset_name = args["dataset_name"]
    
#     if cl_args.name:
#         save_dir = f"{save_dir}_{cl_args.name}_{dataset_name}_{property_name}"
#     else:
#         save_dir = f"{save_dir}_{cl_args.config}_{dataset_name}_{property_name}"
    
#     os.makedirs(save_dir, exist_ok=True)

#     try:
#         if cl_args.bg:
#             # 重定向输出
#             sys.stdout = open(f"{save_dir}/stdout.txt", 'w')
#             sys.stderr = open(f"{save_dir}/stderr.txt", 'w')

#         # 保存配置
#         with open(f"{save_dir}/config.json", 'w') as f:
#             json.dump(args, f, indent=2)
        
#         # 设置设备和随机种子
#         device = torch.device("cpu")#('cuda' if torch.cuda.is_available() else 'cpu')
#         torch.set_default_dtype(torch.float64)
#         # torch.manual_seed(int(args["seed"]))
#         # np.random.seed(int(args["seed"]))

#         # 处理数据集路径
#         data_dir = args.get("data_dir", "./data")
#         dataset_path = os.path.join(data_dir, dataset_name)
        
#         # 检查是否需要解压
#         if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
#             print(f"解压数据文件: {dataset_path}.gz")
#             with gzip.open(dataset_path + ".gz", 'rb') as f_in:
#                 with open(dataset_path, 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
        
#         # 如果仍然找不到文件，添加.csv后缀再尝试
#         if not os.path.exists(dataset_path):
#             dataset_path = dataset_path + ".csv"
#             if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
#                 print(f"解压数据文件: {dataset_path}.gz")
#                 with gzip.open(dataset_path + ".gz", 'rb') as f_in:
#                     with open(dataset_path, 'wb') as f_out:
#                         shutil.copyfileobj(f_in, f_out)
        
#         if not os.path.exists(dataset_path):
#             raise FileNotFoundError(f"找不到数据集文件: {dataset_path} 或 {dataset_path}.gz")

#         # 创建优化问题
#         print(f"创建SMILES离散优化问题...")
#         problem = SMILESDiscreteProblem(
#             smiles_data_path=dataset_path,
#             property_name=property_name,
#             feature_type=args.get("feature_type", "MorganGenerator"),
#             fp_radius=args.get("fp_radius", 2),
#             fp_bits=args.get("fp_bits", 60),
#             discretize=args.get("discretize", True),
#             n_bins=args.get("n_bins", 2),
#             maximize=args.get("maximize", False),
#             seed=int(args["seed"])
#         )
        
#         # 保存问题定义
#         problem.save(os.path.join(save_dir, "problem.pkl"))
        
#         # 获取维度
#         input_dim = problem.dim
#         output_dim = 1  # 单目标优化
        
#         print(f"问题维度: {input_dim}")
#         print(f"特征类型: {problem.feature_type}")
#         print(f"离散化: {problem.discretize}")
#         print(f"目标: {'最大化' if problem.maximize else '最小化'} {property_name}")

#         # 开始多次实验
#         for trial in range(1, args["n_trials"] + 1):
#             print("-" * 20, f"开始实验 {trial}" + "-" * 20)
            
#             # 采样初始点
#             n_init_points = args["n_init_points"]
#             init_x, init_y = problem.sample_sobol_points(n_init_points)
#             init_x = init_x.to(device)
#             init_y = init_y.to(device)
            
#             # 为每个模型运行贝叶斯优化
#             model_dict = args["models"]
#             for model_id, model_args in model_dict.items():
#                 model_name = model_args["model"]

#                 model_save_dir = f"{save_dir}/trial_{trial}/{model_id}"
#                 os.makedirs(model_save_dir, exist_ok=True)
#                 os.makedirs(f"{model_save_dir}/model_state", exist_ok=True)
#                 os.makedirs(f"{model_save_dir}/queries", exist_ok=True)

#                 print("-" * 20, f"运行 {model_id}" + "-" * 20)
#                 start_time = time.time()
                
#                 model = initialize_model(model_name, model_args, input_dim, output_dim, device)
#                 best_x, best_y, best_smiles = bayes_opt_molecular(
#                     model, problem, args, init_x, init_y, model_save_dir, device, model_name
#                 )
#                 del model

#                 print(f"\n找到的最佳值: {best_y.item()}")
#                 print(f"最佳分子: {best_smiles}")
#                 print(f"用时(秒): {time.time() - start_time:.2f}")

#             torch.cuda.empty_cache()

#         os.rename(save_dir, save_dir + "_done")
#         print("优化完成!")
    
#     except Exception as e:
#         print(traceback.format_exc())
#         os.rename(save_dir, save_dir + "_canceled")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="分子优化程序")
#     parser.add_argument("--config", type=str, default="molecular_default", help="配置文件名")
#     parser.add_argument("--bg", action="store_true", help="是否在后台运行")
#     parser.add_argument("-n", "--name", type=str, help="实验名称(可选)")
#     cl_args = parser.parse_args()

#     main(cl_args)


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
from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement #qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import normalize, unnormalize
from models import *
import wandb
import numpy as np

from smiles_discrete_problem import SMILESDiscreteProblem

def round_feature_vector(x, problem):
    """根据问题类型对特征向量取整"""
    if problem.discretize:
        return torch.round(x)
    return x

def bayes_opt_molecular(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
    """
    针对分子优化的贝叶斯优化函数
    
    参数:
    ----
    model: 模型实例
    problem: SMILESDiscreteProblem实例
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

    # 记录对应的SMILES
    train_smiles = []
    for i in range(len(train_x)):
        smiles = problem.get_smiles_from_feature(train_x[i].cpu().numpy())
        train_smiles.append(smiles)
    
    # 保存初始分子集
    with open(f"{model_save_dir}/initial_molecules.csv", "w") as f:
        f.write("SMILES," + args["property_name"] + "\n")
        for smiles, y in zip(train_smiles, train_y):
            f.write(f"{smiles},{y.item()}\n")

    # 使用wandb跟踪实验
    use_wandb = args.get("use_wandb", True)
    if use_wandb:
        wandb_project = args.get("wandb_project", "Molecular_Optimization")
        wandb.init(
            project=wandb_project,
            name=f"{model_name}-{args['property_name']}-{args.get('acquisition_function', 'ucb').upper()}",
            config=args
        )

    # 主优化循环
    for i in range(args["n_BO_iters"]):
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

        # 在归一化的特征空间上拟合模型
        model_start = time.time()
        normalized_x = train_x#normalize(train_x, bounds).to(train_x)
        model.fit_and_save(normalized_x, train_y, model_save_dir)
        model_end = time.time()
        print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
        # 构建采集函数
        acq_start = time.time()
        sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
        # 根据配置选择采集函数类型
        acq_type = args.get("acquisition_function", "ucb").lower()
        
        if acq_type == "ucb":
            # 使用Upper Confidence Bound作为采集函数
            beta = args.get("ucb_beta", 2.0)  # 默认值为2.0
            acquisition = qUpperConfidenceBound(
                model=model,
                beta=beta,
                sampler=sampler
            )
            print(f"使用UCB采集函数，beta={beta}")
            
        elif acq_type == "ei":
            # 使用Expected Improvement作为采集函数
            best_f = train_y.max() if problem.maximize else -train_y.min()
            acquisition = qExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler
            )
            print(f"使用EI采集函数，best_f={best_f.item()}")
            
        elif acq_type == "logei":
            # 使用Log Expected Improvement作为采集函数
            best_f = train_y.max() if problem.maximize else -train_y.min()
            acquisition = qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler
            )
            print(f"使用LogEI采集函数，best_f={best_f.item()}")
            
        else:
            raise ValueError(f"不支持的采集函数类型: {acq_type}. 支持的类型: ucb, ei, logei")
        
        # 优化采集函数
        normalized_candidates, acqf_values = optimize_acqf(
            acquisition, standard_bounds, q=q, num_restarts=2, raw_samples=16, return_best_only=False,
            options={"batch_limit": 1, "maxiter": 10})
        candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)

        # 对候选点取整
        candidates = round_feature_vector(candidates, problem)
        
        # 计算取整后的采集值
        normalized_rounded_candidates = normalize(candidates, bounds)
        acqf_values = acquisition(normalized_rounded_candidates)
        acq_end = time.time()
        print(f"采集函数优化时间: {acq_end - acq_start:.2f}秒")

        # 选择最佳候选点
        best_index = acqf_values.max(dim=0).indices.item()
        new_x = candidates[best_index].to(train_x)

        # 释放内存
        del acquisition
        del acqf_values
        del normalized_candidates
        del normalized_rounded_candidates
        torch.cuda.empty_cache()

        # 评估新点
        new_y = problem(new_x)
        # 添加输出维度
        if output_dim == 1:
            new_y = new_y.unsqueeze(-1)
        
        # 获取对应的SMILES
        new_smiles = problem.get_smiles_from_feature(new_x.cpu().numpy())
        train_smiles.append(new_smiles)
        
        # 扩展训练集
        train_x = torch.cat([train_x, new_x])#.unsqueeze(0)])
        train_y = torch.cat([train_y, new_y])
        
        # 打印当前最佳值
        if problem.maximize:
            best_idx = torch.argmax(train_y)
            best_val = train_y[best_idx].item()
            best_smiles = train_smiles[best_idx]
        else:
            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx].item()
            best_smiles = train_smiles[best_idx]
        
        print(f"新评估的分子: {new_smiles}")
        print(f"新评估的值: {new_y.item() if output_dim == 1 else new_y}")
        print(f"当前最佳值: {best_val} (对应分子: {best_smiles})")
        
        # 记录到wandb
        if use_wandb:
            log_data = {
                "iteration": i + 1,
                "best_value": best_val,
                "actual_value": new_y.item() if output_dim == 1 else new_y[0].item(),
                "best_molecule": best_smiles,
                "acquisition_function": acq_type
            }
            
            # 根据采集函数类型添加特定参数
            if acq_type == "ucb":
                log_data["ucb_beta"] = beta
            elif acq_type in ["ei", "logei"]:
                log_data["best_f"] = best_f.item()
                
            wandb.log(log_data)

    # 保存结果
    if model_save_dir is not None:
        torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
        torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
        # 保存所有评估过的分子
        with open(f"{model_save_dir}/evaluated_molecules.csv", "w") as f:
            f.write("SMILES," + args["property_name"] + "\n")
            for smiles, y in zip(train_smiles, train_y):
                f.write(f"{smiles},{y.item()}\n")
        
        # 保存最佳分子
        if problem.maximize:
            best_idx = torch.argmax(train_y)
            best_val = train_y[best_idx].item()
        else:
            best_idx = torch.argmin(train_y)
            best_val = train_y[best_idx].item()
        
        best_x = train_x[best_idx]
        best_smiles = train_smiles[best_idx]
        
        with open(f"{model_save_dir}/best_molecule.txt", "w") as f:
            f.write(f"SMILES: {best_smiles}\n")
            f.write(f"{args['property_name']}: {best_val}\n")
            f.write(f"Acquisition Function: {args.get('acquisition_function', 'ucb').upper()}\n")
            if args.get('acquisition_function', 'ucb').lower() == 'ucb':
                f.write(f"UCB Beta: {args.get('ucb_beta', 2.0)}\n")

    if use_wandb:
        wandb.finish()

    return best_x, train_y[best_idx], best_smiles


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
    """主函数"""
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
        data_dir = args.get("data_dir", "./data")
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # 检查是否需要解压
        if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
            print(f"解压数据文件: {dataset_path}.gz")
            with gzip.open(dataset_path + ".gz", 'rb') as f_in:
                with open(dataset_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        # 如果仍然找不到文件，添加.csv后缀再尝试
        if not os.path.exists(dataset_path):
            dataset_path = dataset_path + ".csv"
            if not os.path.exists(dataset_path) and os.path.exists(dataset_path + ".gz"):
                print(f"解压数据文件: {dataset_path}.gz")
                with gzip.open(dataset_path + ".gz", 'rb') as f_in:
                    with open(dataset_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到数据集文件: {dataset_path} 或 {dataset_path}.gz")

        # 创建优化问题
        print(f"创建SMILES离散优化问题...")
        problem = SMILESDiscreteProblem(
            smiles_data_path=dataset_path,
            property_name=property_name,
            feature_type=args.get("feature_type", "MorganGenerator"),
            fp_radius=args.get("fp_radius", 2),
            fp_bits=args.get("fp_bits", 60),
            discretize=args.get("discretize", True),
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
        print(f"特征类型: {problem.feature_type}")
        print(f"离散化: {problem.discretize}")
        print(f"目标: {'最大化' if problem.maximize else '最小化'} {property_name}")
        print(f"采集函数: {args.get('acquisition_function', 'ucb').upper()}")
        if args.get('acquisition_function', 'ucb').lower() == 'ucb':
            print(f"UCB Beta参数: {args.get('ucb_beta', 2.0)}")

        # 开始多次实验
        for trial in range(1, args["n_trials"] + 1):
            print("-" * 20, f"开始实验 {trial}" + "-" * 20)
            
            # 采样初始点
            n_init_points = args["n_init_points"]
            init_x, init_y = problem.sample_sobol_points(n_init_points)
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
                best_x, best_y, best_smiles = bayes_opt_molecular(
                    model, problem, args, init_x, init_y, model_save_dir, device, model_name
                )
                del model

                print(f"\n找到的最佳值: {best_y.item()}")
                print(f"最佳分子: {best_smiles}")
                print(f"用时(秒): {time.time() - start_time:.2f}")

            torch.cuda.empty_cache()

        os.rename(save_dir, save_dir + "_done")
        print("优化完成!")
    
    except Exception as e:
        print(traceback.format_exc())
        os.rename(save_dir, save_dir + "_canceled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分子优化程序")
    parser.add_argument("--config", type=str, default="molecular_default", help="配置文件名")
    parser.add_argument("--bg", action="store_true", help="是否在后台运行")
    parser.add_argument("-n", "--name", type=str, help="实验名称(可选)")
    cl_args = parser.parse_args()

    main(cl_args)