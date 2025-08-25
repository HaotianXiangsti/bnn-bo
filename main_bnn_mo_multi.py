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
# from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
# from botorch.optim import optimize_acqf
# from botorch.sampling.stochastic_samplers import StochasticSampler
# from botorch.utils.transforms import normalize, unnormalize
# from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
# from botorch.utils.multi_objective.pareto import is_non_dominated
# from models import *
# import wandb
# import numpy as np

# from smiles_discrete_problem import SMILESDiscreteProblemMulti

# def round_feature_vector(x, problem):
#     """根据问题类型对特征向量取整"""
#     if problem.discretize:
#         return torch.round(x)
#     return x

# def get_pareto_front(Y):
#     """获取帕累托前沿"""
#     if Y.dim() == 1:
#         Y = Y.unsqueeze(-1)
    
#     # 使用botorch的is_non_dominated函数
#     pareto_mask = is_non_dominated(Y)
#     return Y[pareto_mask], pareto_mask

# def compute_hypervolume_indicator(Y, ref_point):
#     """计算超体积指标"""
#     from botorch.utils.multi_objective.hypervolume import Hypervolume
    
#     if Y.dim() == 1:
#         Y = Y.unsqueeze(-1)
    
#     # 获取帕累托前沿
#     pareto_Y, _ = get_pareto_front(Y)
    
#     if len(pareto_Y) == 0:
#         return 0.0
    
#     # 计算超体积
#     hv = Hypervolume(ref_point=ref_point)
#     return hv.compute(pareto_Y)

# def bayes_opt_molecular_multi(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
#     """
#     针对多目标分子优化的贝叶斯优化函数
    
#     参数:
#     ----
#     model: 模型实例
#     problem: SMILESDiscreteProblemMulti实例
#     args: 配置参数
#     init_x: 初始特征向量
#     init_y: 初始目标值 (shape: [n_points, n_objectives])
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
#     print("初始值形状:", train_y.shape)
#     print("初始值示例:", train_y[:3])

#     # 记录对应的SMILES
#     train_smiles = []
#     for i in range(len(train_x)):
#         smiles = problem.get_smiles_from_feature(train_x[i].cpu().numpy())
#         train_smiles.append(smiles)
    
#     # 保存初始分子集
#     objective_names = args["objective_names"]
#     with open(f"{model_save_dir}/initial_molecules.csv", "w") as f:
#         header = "SMILES," + ",".join(objective_names) + "\n"
#         f.write(header)
#         for smiles, y in zip(train_smiles, train_y):
#             values = ",".join([str(val.item()) for val in y])
#             f.write(f"{smiles},{values}\n")

#     # 设置参考点 (用于超体积计算)
#     # 对于最大化问题，参考点应该在最差值之下
#     # 对于最小化问题，参考点应该在最差值之上
#     ref_point = torch.zeros(output_dim).to(train_y)
#     for i, maximize in enumerate(args["maximize"]):
#         if maximize:
#             ref_point[i] = train_y[:, i].min() - 0.1 * (train_y[:, i].max() - train_y[:, i].min())
#         else:
#             #ref_point[i] = train_y[:, i].max() + 0.1 * (train_y[:, i].max() - train_y[:, i].min())
#             ref_point[i] = train_y[:, i].min() - 0.1 * (train_y[:, i].max() - train_y[:, i].min())
    
#     print(f"参考点: {ref_point}")

#     # 使用wandb跟踪实验
#     use_wandb = args.get("use_wandb", True)
#     if use_wandb:
#         wandb_project = args.get("wandb_project", "Molecular_Multi_Optimization")
#         wandb.init(
#             project=wandb_project,
#             name=f"{model_name}-{'-'.join(objective_names)}-{args.get('acquisition_function', 'qehvi').upper()}",
#             config=args
#         )

#     # 主优化循环
#     for i in range(args["n_BO_iters"]):
#         sys.stdout.flush()
#         sys.stderr.flush()
#         print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

#         # 在归一化的特征空间上拟合模型
#         model_start = time.time()
#         normalized_x = train_x
#         model.fit_and_save(normalized_x, train_y, model_save_dir)
#         model_end = time.time()
#         print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
#         # 构建采集函数
#         acq_start = time.time()
#         sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
#         # 根据配置选择采集函数类型
#         acq_type = args.get("acquisition_function", "qehvi").lower()
        
#         # 为每个目标应用最大化/最小化设置
#         # botorch假设所有目标都是最大化的，对于最小化目标需要取反
#         adjusted_train_y = train_y.clone()
#         for j, maximize in enumerate(args["maximize"]):
#             if not maximize:
#                 adjusted_train_y[:, j] = -adjusted_train_y[:, j]
        
#         # 调整参考点
#         adjusted_ref_point = ref_point.clone()
#         for j, maximize in enumerate(args["maximize"]):
#             if not maximize:
#                 adjusted_ref_point[j] = -adjusted_ref_point[j]
        
#         if acq_type == "qehvi":
#             # 使用qExpectedHypervolumeImprovement
#             partitioning = NondominatedPartitioning(
#                 ref_point=adjusted_ref_point,
#                 Y=adjusted_train_y
#             )
            
#             acquisition = qExpectedHypervolumeImprovement(
#                 model=model,
#                 ref_point=adjusted_ref_point,
#                 partitioning=partitioning,
#                 sampler=sampler
#             )
#             print(f"使用qEHVI采集函数")
            
#         elif acq_type == "qnehvi":
#             # 使用qNoisyExpectedHypervolumeImprovement
#             partitioning = NondominatedPartitioning(
#                 ref_point=adjusted_ref_point,
#                 Y=adjusted_train_y
#             )
            
#             acquisition = qNoisyExpectedHypervolumeImprovement(
#                 model=model,
#                 ref_point=adjusted_ref_point,
#                 X_baseline=normalized_x,
#                 sampler=sampler,
#                 prune_baseline=True
#             )
#             print(f"使用qNEHVI采集函数")
            
#         else:
#             raise ValueError(f"不支持的采集函数类型: {acq_type}. 支持的类型: qehvi, qnehvi")
        
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
#         # 确保输出维度正确
#         if new_y.dim() == 1:
#             new_y = new_y.unsqueeze(0)
        
#         # 获取对应的SMILES
#         new_smiles = problem.get_smiles_from_feature(new_x.cpu().numpy())
#         train_smiles.append(new_smiles)
        
#         # 扩展训练集
#         train_x = torch.cat([train_x, new_x])#new_x.unsqueeze(0)])
#         train_y = torch.cat([train_y, new_y])
        
#         # 计算当前帕累托前沿
#         pareto_y, pareto_mask = get_pareto_front(train_y)
#         pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if pareto_mask[j]]
        
#         # 计算超体积指标
#         hv = compute_hypervolume_indicator(train_y, ref_point)
        
#         print(f"新评估的分子: {new_smiles}")
#         print(f"新评估的值: {new_y.squeeze()}")
#         print(f"当前帕累托前沿大小: {len(pareto_y)}")
#         print(f"当前超体积: {hv:.4f}")
        
#         # 记录到wandb
#         if use_wandb:
#             log_data = {
#                 "iteration": i + 1,
#                 "pareto_front_size": len(pareto_y),
#                 "hypervolume": hv,
#                 "acquisition_function": acq_type
#             }
            
#             # 记录每个目标的新值
#             for j, obj_name in enumerate(objective_names):
#                 log_data[f"new_{obj_name}"] = new_y.squeeze()[j].item()
            
#             # 记录帕累托前沿的统计信息
#             for j, obj_name in enumerate(objective_names):
#                 if len(pareto_y) > 0:
#                     log_data[f"pareto_{obj_name}_mean"] = pareto_y[:, j].mean().item()
#                     log_data[f"pareto_{obj_name}_std"] = pareto_y[:, j].std().item()
#                     log_data[f"pareto_{obj_name}_min"] = pareto_y[:, j].min().item()
#                     log_data[f"pareto_{obj_name}_max"] = pareto_y[:, j].max().item()
                
#             wandb.log(log_data)

#     # 保存结果
#     if model_save_dir is not None:
#         torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
#         torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
#         # 保存所有评估过的分子
#         with open(f"{model_save_dir}/evaluated_molecules.csv", "w") as f:
#             header = "SMILES," + ",".join(objective_names) + "\n"
#             f.write(header)
#             for smiles, y in zip(train_smiles, train_y):
#                 values = ",".join([str(val.item()) for val in y])
#                 f.write(f"{smiles},{values}\n")
        
#         # 保存帕累托前沿
#         final_pareto_y, final_pareto_mask = get_pareto_front(train_y)
#         final_pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if final_pareto_mask[j]]
        
#         with open(f"{model_save_dir}/pareto_front.csv", "w") as f:
#             header = "SMILES," + ",".join(objective_names) + "\n"
#             f.write(header)
#             for smiles, y in zip(final_pareto_smiles, final_pareto_y):
#                 values = ",".join([str(val.item()) for val in y])
#                 f.write(f"{smiles},{values}\n")
        
#         # 保存优化摘要
#         final_hv = compute_hypervolume_indicator(train_y, ref_point)
#         with open(f"{model_save_dir}/optimization_summary.txt", "w") as f:
#             f.write(f"多目标优化摘要\n")
#             f.write(f"目标: {', '.join(objective_names)}\n")
#             f.write(f"最大化/最小化: {args['maximize']}\n")
#             f.write(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}\n")
#             f.write(f"总评估次数: {len(train_y)}\n")
#             f.write(f"最终帕累托前沿大小: {len(final_pareto_y)}\n")
#             f.write(f"最终超体积: {final_hv:.4f}\n")
#             f.write(f"参考点: {ref_point.tolist()}\n")

#     if use_wandb:
#         wandb.finish()

#     return train_x, train_y, final_pareto_y, final_pareto_smiles


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
#     objective_names = args["objective_names"]
#     dataset_name = args["dataset_name"]
    
#     objectives_str = "-".join(objective_names)
#     if cl_args.name:
#         save_dir = f"{save_dir}_{cl_args.name}_{dataset_name}_{objectives_str}"
#     else:
#         save_dir = f"{save_dir}_{cl_args.config}_{dataset_name}_{objectives_str}"
    
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
#         device = torch.device("cpu")
#         torch.set_default_dtype(torch.float64)

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

#         # 创建多目标优化问题
#         print(f"创建多目标SMILES离散优化问题...")
#         problem = SMILESDiscreteProblemMulti(
#             smiles_data_path=dataset_path,
#             objective_names=objective_names,
#             feature_type=args.get("feature_type", "morgan"),
#             fp_radius=args.get("fp_radius", 2),
#             fp_bits=args.get("fp_bits", 60),
#             discretize=args.get("discretize", True),
#             n_bins=args.get("n_bins", 2),
#             maximize=args.get("maximize", [True] * len(objective_names)),
#             seed=int(args["seed"])
#         )
        
#         # 保存问题定义
#         problem.save(os.path.join(save_dir, "problem.pkl"))
        
#         # 获取维度
#         input_dim = problem.dim
#         output_dim = len(objective_names)
        
#         print(f"问题维度: {input_dim}")
#         print(f"目标数量: {output_dim}")
#         print(f"目标名称: {objective_names}")
#         print(f"最大化/最小化: {args['maximize']}")
#         print(f"特征类型: {problem.feature_type}")
#         print(f"离散化: {problem.discretize}")
#         print(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}")

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
#                 train_x, train_y, pareto_y, pareto_smiles = bayes_opt_molecular_multi(
#                     model, problem, args, init_x, init_y, model_save_dir, device, model_name
#                 )
#                 del model

#                 print(f"\n找到的帕累托前沿大小: {len(pareto_y)}")
#                 print(f"帕累托前沿示例:")
#                 for i, (smiles, y) in enumerate(zip(pareto_smiles[:3], pareto_y[:3])):
#                     print(f"  {i+1}. {smiles}: {y}")
#                 print(f"用时(秒): {time.time() - start_time:.2f}")

#             torch.cuda.empty_cache()

#         os.rename(save_dir, save_dir + "_done")
#         print("多目标优化完成!")
    
#     except Exception as e:
#         print(traceback.format_exc())
#         os.rename(save_dir, save_dir + "_canceled")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="多目标分子优化程序")
#     parser.add_argument("--config", type=str, default="molecular_multi_default", help="配置文件名")
#     parser.add_argument("--bg", action="store_true", help="是否在后台运行")
#     parser.add_argument("-n", "--name", type=str, help="实验名称(可选)")
#     cl_args = parser.parse_args()

#     main(cl_args)


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
# from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
# from botorch.optim import optimize_acqf
# from botorch.sampling.stochastic_samplers import StochasticSampler
# from botorch.utils.transforms import normalize, unnormalize
# from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
# from botorch.utils.multi_objective.pareto import is_non_dominated
# from models import *
# import wandb
# import numpy as np

# from smiles_discrete_problem import SMILESDiscreteProblemMulti

# def round_feature_vector(x, problem):
#     """根据问题类型对特征向量取整"""
#     if problem.discretize:
#         return torch.round(x)
#     return x

# def get_pareto_front(Y):
#     """获取帕累托前沿"""
#     if Y.dim() == 1:
#         Y = Y.unsqueeze(-1)
    
#     # 使用botorch的is_non_dominated函数
#     pareto_mask = is_non_dominated(Y)
#     return Y[pareto_mask], pareto_mask

# def compute_hypervolume_indicator(Y, ref_point):
#     """计算超体积指标"""
#     from botorch.utils.multi_objective.hypervolume import Hypervolume
    
#     if Y.dim() == 1:
#         Y = Y.unsqueeze(-1)
    
#     # 获取帕累托前沿
#     pareto_Y, _ = get_pareto_front(Y)
    
#     if len(pareto_Y) == 0:
#         return 0.0
    
#     # 计算超体积
#     hv = Hypervolume(ref_point=ref_point)
#     return hv.compute(pareto_Y)

# def compute_reference_hypervolume(problem, ref_point):
#     """
#     计算数据集中帕累托前沿的参考超体积
    
#     参数:
#     ----
#     problem: SMILESDiscreteProblemMulti实例
#     ref_point: 参考点
    
#     返回:
#     ----
#     参考超体积值
#     """
#     from botorch.utils.multi_objective.hypervolume import Hypervolume
    
#     # 获取数据集中的帕累托前沿
#     _, pareto_values, _ = problem.get_pareto_front_from_data()
    
#     if len(pareto_values) == 0:
#         print("未找到帕累托前沿")
#         return 0.0
    
#     # 转换为张量
#     pareto_Y = torch.tensor(pareto_values, dtype=torch.float64)

#     adjusted_pareto_Y = pareto_Y.clone()
    
#     #根据最大化/最小化设置调整目标值（与主优化循环中的处理保持一致）
#     adjusted_pareto_Y = pareto_Y.clone()
#     for j, maximize in enumerate(problem.maximize):
#         if not maximize:
#             adjusted_pareto_Y[:, j] = -adjusted_pareto_Y[:, j]
    
#     # 计算参考超体积
#     hv = Hypervolume(ref_point=ref_point)
#     print(f"计算参考超体积，参考点: {ref_point}, 帕累托前沿大小: {len(adjusted_pareto_Y)}")
#     return hv.compute(adjusted_pareto_Y)

# def compute_log_hypervolume_difference(current_hv, reference_hv):
#     """
#     计算Log Hypervolume Difference (LHD)指标
    
#     参数:
#     ----
#     current_hv: 当前超体积
#     reference_hv: 参考超体积（通常是数据集帕累托前沿的超体积）
    
#     返回:
#     ----
#     LHD值：log10(reference_hv - current_hv + epsilon)
#     """
#     # 添加小的epsilon防止取对数时出现负数或零
#     epsilon = 1e-10
#     print(f"当前超体积: {current_hv}, 参考超体积: {reference_hv},")
#     # 确保参考超体积大于当前超体积
#     # if reference_hv <= current_hv:
#     #     # 如果当前超体积已经达到或超过参考值，返回一个很小的负值
#     #     return -10.0  # 对应log10(1e-10)
    
#     difference = reference_hv - current_hv + epsilon

#     print(f"超体积差值: {difference}, epsilon: {epsilon}")
#     return np.log10(difference)

# def bayes_opt_molecular_multi(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
#     """
#     针对多目标分子优化的贝叶斯优化函数
    
#     参数:
#     ----
#     model: 模型实例
#     problem: SMILESDiscreteProblemMulti实例
#     args: 配置参数
#     init_x: 初始特征向量
#     init_y: 初始目标值 (shape: [n_points, n_objectives])
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
#     print("初始值形状:", train_y.shape)
#     print("初始值示例:", train_y[:3])

#     # 记录对应的SMILES
#     train_smiles = []
#     for i in range(len(train_x)):
#         smiles = problem.get_smiles_from_feature(train_x[i].cpu().numpy())
#         train_smiles.append(smiles)
    
#     # 保存初始分子集
#     objective_names = args["objective_names"]
#     with open(f"{model_save_dir}/initial_molecules.csv", "w") as f:
#         header = "SMILES," + ",".join(objective_names) + "\n"
#         f.write(header)
#         for smiles, y in zip(train_smiles, train_y):
#             values = ",".join([str(val.item()) for val in y])
#             f.write(f"{smiles},{values}\n")

#     # 设置参考点 (用于超体积计算)
#     # 对于最大化问题，参考点应该在最差值之下
#     # 对于最小化问题，参考点应该在最差值之上
#     # ref_point = torch.zeros(output_dim).to(train_y)
#     # for i, maximize in enumerate(args["maximize"]):
#     #     if maximize:
#     #         ref_point[i] = train_y[:, i].min() - 0.1 * (train_y[:, i].max() - train_y[:, i].min())
#     #     else:
#     #         ref_point[i] = train_y[:, i].min() - 0.1 * (train_y[:, i].max() - train_y[:, i].min())
#     #         #ref_point[i] = train_y[:, i].max() + 0.1 * (train_y[:, i].max() - train_y[:, i].min())

#     # print(f"参考点: {ref_point}")

#     # ========== 修正：基于整个数据集设置参考点 ==========
#     # 获取整个数据集的目标值（而不是仅初始采样点）
#     all_Y = torch.tensor(problem.y, dtype=torch.float64, device=train_y.device)
    
#     print("\n整个数据集的目标值统计:")
#     for i, obj_name in enumerate(objective_names):
#         print(f"  {obj_name}: min={all_Y[:, i].min():.4f}, max={all_Y[:, i].max():.4f}, "
#               f"mean={all_Y[:, i].mean():.4f}, std={all_Y[:, i].std():.4f}")
    
#     # 设置参考点（用于超体积计算）
#     # botorch假设所有目标都是最大化的，所以我们需要：
#     # 1. 对于原始最大化目标：参考点应该在最小值之下
#     # 2. 对于原始最小化目标：我们会取反，所以参考点应该在-max之下
    
#     ref_point = torch.zeros(output_dim, dtype=torch.float64, device=train_y.device)
    
#     for i, maximize in enumerate(args["maximize"]):
#         if maximize:
#             # 对于最大化目标，参考点设在数据集最小值之下
#             min_val = all_Y[:, i].min()
#             range_val = all_Y[:, i].max() - all_Y[:, i].min()
#             # 参考点 = 最小值 - 10% * 范围
#             ref_point[i] = min_val - 0.1 * range_val
#             print(f"  {objective_names[i]} (最大化): 参考点={ref_point[i]:.4f} < 最小值={min_val:.4f}")
#         else:
#             # 对于最小化目标，因为会取反，所以参考点设在-max之下
#             max_val = all_Y[:, i].max()
#             range_val = all_Y[:, i].max() - all_Y[:, i].min()
#             # 参考点 = -最大值 - 10% * 范围
#             ref_point[i] = -max_val - 0.1 * range_val
#             print(f"  {objective_names[i]} (最小化): 参考点={ref_point[i]:.4f} < -最大值={-max_val:.4f}")

#     print(f"\n最终参考点: {ref_point}")
    
#     # 计算参考超体积（数据集帕累托前沿的超体积）
#     # 需要先调整参考点以适应最大化/最小化设置
#     # adjusted_ref_point = ref_point.clone()
#     # for j, maximize in enumerate(args["maximize"]):
#     #     if not maximize:
#     #         adjusted_ref_point[j] = -adjusted_ref_point[j]
#     adjusted_ref_point = ref_point.clone()
#     try:
#         reference_hv = compute_reference_hypervolume(problem, adjusted_ref_point)
#         print(f"参考超体积（数据集帕累托前沿）: {reference_hv:.6f}")
#     except Exception as e:
#         print(f"警告: 无法计算参考超体积，将使用默认值: {e}")
#         reference_hv = 1.0  # 使用默认值

#     # adjusted_ref_point = ref_point.clone()
#     # for j, maximize in enumerate(args["maximize"]):
#     #     if not maximize:
#     #         adjusted_ref_point[j] = -adjusted_ref_point[j]

#     # 使用wandb跟踪实验
#     use_wandb = args.get("use_wandb", True)
#     if use_wandb:
#         wandb_project = args.get("wandb_project", "Molecular_Multi_Optimization")
#         wandb.init(
#             project=wandb_project,
#             name=f"{model_name}-{'-'.join(objective_names)}-{args.get('acquisition_function', 'qehvi').upper()}",
#             config=args
#         )
        
#         # 记录参考超体积
#         wandb.log({"reference_hypervolume": reference_hv}, step=0)

#     # 主优化循环
#     for i in range(args["n_BO_iters"]):
#         sys.stdout.flush()
#         sys.stderr.flush()
#         print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

#         # 在归一化的特征空间上拟合模型
#         model_start = time.time()
#         normalized_x = train_x
#         model.fit_and_save(normalized_x, train_y, model_save_dir)
#         model_end = time.time()
#         print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
#         # 构建采集函数
#         acq_start = time.time()
#         sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
#         # 根据配置选择采集函数类型
#         acq_type = args.get("acquisition_function", "qehvi").lower()
        
#         # 为每个目标应用最大化/最小化设置
#         # botorch假设所有目标都是最大化的，对于最小化目标需要取反
#         # adjusted_train_y = train_y.clone()
#         # for j, maximize in enumerate(args["maximize"]):
#         #     if not maximize:
#         #         adjusted_train_y[:, j] = -adjusted_train_y[:, j]

#         adjusted_train_y = train_y.clone()
        
#         # 调整参考点
#         #adjusted_ref_point = ref_point.clone()
#         # for j, maximize in enumerate(args["maximize"]):
#         #     if not maximize:
#         #         adjusted_ref_point[j] = -adjusted_ref_point[j]
        
#         if acq_type == "qehvi":
#             # 使用qExpectedHypervolumeImprovement
#             partitioning = NondominatedPartitioning(
#                 ref_point=adjusted_ref_point,
#                 Y=adjusted_train_y
#             )
            
#             acquisition = qExpectedHypervolumeImprovement(
#                 model=model,
#                 ref_point=adjusted_ref_point,
#                 partitioning=partitioning,
#                 sampler=sampler
#             )
#             print(f"使用qEHVI采集函数")
            
#         elif acq_type == "qnehvi":
#             # 使用qNoisyExpectedHypervolumeImprovement
#             partitioning = NondominatedPartitioning(
#                 ref_point=adjusted_ref_point,
#                 Y=adjusted_train_y
#             )
            
#             acquisition = qNoisyExpectedHypervolumeImprovement(
#                 model=model,
#                 ref_point=adjusted_ref_point,
#                 X_baseline=normalized_x,
#                 sampler=sampler,
#                 prune_baseline=True
#             )
#             print(f"使用qNEHVI采集函数")
            
#         else:
#             raise ValueError(f"不支持的采集函数类型: {acq_type}. 支持的类型: qehvi, qnehvi")
        
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
#         # 确保输出维度正确
#         if new_y.dim() == 1:
#             new_y = new_y.unsqueeze(0)
        
#         # 获取对应的SMILES
#         new_smiles = problem.get_smiles_from_feature(new_x.cpu().numpy())
#         train_smiles.append(new_smiles)
        
#         # 扩展训练集
#         train_x = torch.cat([train_x, new_x])
#         train_y = torch.cat([train_y, new_y])
        
#         # 计算当前帕累托前沿
#         pareto_y, pareto_mask = get_pareto_front(train_y)
#         pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if pareto_mask[j]]
        
#         # 计算超体积指标
#         hv = compute_hypervolume_indicator(train_y, ref_point)
        
#         # 计算Log Hypervolume Difference (LHD)
#         lhd = compute_log_hypervolume_difference(hv, reference_hv)
        
#         print(f"新评估的分子: {new_smiles}")
#         print(f"新评估的值: {new_y.squeeze()}")
#         print(f"当前帕累托前沿大小: {len(pareto_y)}")
#         print(f"当前超体积: {hv:.6f}")
#         print(f"Log Hypervolume Difference (LHD): {lhd:.6f}")
        
#         # 记录到wandb
#         if use_wandb:
#             log_data = {
#                 "iteration": i + 1,
#                 "pareto_front_size": len(pareto_y),
#                 "hypervolume": hv,
#                 "log_hypervolume_difference": lhd,
#                 "acquisition_function": acq_type
#             }
            
#             # 记录每个目标的新值
#             for j, obj_name in enumerate(objective_names):
#                 log_data[f"new_{obj_name}"] = new_y.squeeze()[j].item()
            
#             # 记录帕累托前沿的统计信息
#             for j, obj_name in enumerate(objective_names):
#                 if len(pareto_y) > 0:
#                     log_data[f"pareto_{obj_name}_mean"] = pareto_y[:, j].mean().item()
#                     log_data[f"pareto_{obj_name}_std"] = pareto_y[:, j].std().item()
#                     log_data[f"pareto_{obj_name}_min"] = pareto_y[:, j].min().item()
#                     log_data[f"pareto_{obj_name}_max"] = pareto_y[:, j].max().item()
                
#             wandb.log(log_data)

#     # 保存结果
#     if model_save_dir is not None:
#         torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
#         torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
#         # 保存所有评估过的分子
#         with open(f"{model_save_dir}/evaluated_molecules.csv", "w") as f:
#             header = "SMILES," + ",".join(objective_names) + "\n"
#             f.write(header)
#             for smiles, y in zip(train_smiles, train_y):
#                 values = ",".join([str(val.item()) for val in y])
#                 f.write(f"{smiles},{values}\n")
        
#         # 保存帕累托前沿
#         final_pareto_y, final_pareto_mask = get_pareto_front(train_y)
#         final_pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if final_pareto_mask[j]]
        
#         with open(f"{model_save_dir}/pareto_front.csv", "w") as f:
#             header = "SMILES," + ",".join(objective_names) + "\n"
#             f.write(header)
#             for smiles, y in zip(final_pareto_smiles, final_pareto_y):
#                 values = ",".join([str(val.item()) for val in y])
#                 f.write(f"{smiles},{values}\n")
        
#         # 保存优化摘要
#         final_hv = compute_hypervolume_indicator(train_y, ref_point)
#         final_lhd = compute_log_hypervolume_difference(final_hv, reference_hv)
        
#         with open(f"{model_save_dir}/optimization_summary.txt", "w") as f:
#             f.write(f"多目标优化摘要\n")
#             f.write(f"目标: {', '.join(objective_names)}\n")
#             f.write(f"最大化/最小化: {args['maximize']}\n")
#             f.write(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}\n")
#             f.write(f"总评估次数: {len(train_y)}\n")
#             f.write(f"最终帕累托前沿大小: {len(final_pareto_y)}\n")
#             f.write(f"最终超体积: {final_hv:.6f}\n")
#             f.write(f"参考超体积: {reference_hv:.6f}\n")
#             f.write(f"最终Log Hypervolume Difference (LHD): {final_lhd:.6f}\n")
#             f.write(f"参考点: {ref_point.tolist()}\n")
        
#         # 保存LHD历史记录
#         lhd_history = []
#         for step in range(len(train_y) - len(init_y) + 1):  # 包括初始状态
#             if step == 0:
#                 # 初始状态
#                 current_hv_step = compute_hypervolume_indicator(init_y, ref_point)
#             else:
#                 # 第step次迭代后的状态
#                 current_train_y = torch.cat([init_y, train_y[len(init_y):len(init_y)+step]])
#                 current_hv_step = compute_hypervolume_indicator(current_train_y, ref_point)
            
#             lhd_step = compute_log_hypervolume_difference(current_hv_step, reference_hv)
#             lhd_history.append({
#                 "iteration": step,
#                 "hypervolume": current_hv_step,
#                 "lhd": lhd_step
#             })
        
#         # 保存LHD历史为CSV
#         import pandas as pd
#         lhd_df = pd.DataFrame(lhd_history)
#         lhd_df.to_csv(f"{model_save_dir}/lhd_history.csv", index=False)

#     if use_wandb:
#         # 记录最终的LHD值
#         wandb.log({"final_lhd": final_lhd})
#         wandb.finish()

#     return train_x, train_y, final_pareto_y, final_pareto_smiles


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
#     objective_names = args["objective_names"]
#     dataset_name = args["dataset_name"]
    
#     objectives_str = "-".join(objective_names)
#     if cl_args.name:
#         save_dir = f"{save_dir}_{cl_args.name}_{dataset_name}_{objectives_str}"
#     else:
#         save_dir = f"{save_dir}_{cl_args.config}_{dataset_name}_{objectives_str}"
    
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
#         device = torch.device("cpu")
#         torch.set_default_dtype(torch.float64)

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

#         # 创建多目标优化问题
#         print(f"创建多目标SMILES离散优化问题...")
#         problem = SMILESDiscreteProblemMulti(
#             smiles_data_path=dataset_path,
#             objective_names=objective_names,
#             feature_type=args.get("feature_type", "morgan"),
#             fp_radius=args.get("fp_radius", 2),
#             fp_bits=args.get("fp_bits", 60),
#             discretize=args.get("discretize", True),
#             n_bins=args.get("n_bins", 2),
#             maximize=args.get("maximize", [True] * len(objective_names)),
#             seed=int(args["seed"])
#         )
        
#         # 保存问题定义
#         problem.save(os.path.join(save_dir, "problem.pkl"))
        
#         # 获取维度
#         input_dim = problem.dim
#         output_dim = len(objective_names)
        
#         print(f"问题维度: {input_dim}")
#         print(f"目标数量: {output_dim}")
#         print(f"目标名称: {objective_names}")
#         print(f"最大化/最小化: {args['maximize']}")
#         print(f"特征类型: {problem.feature_type}")
#         print(f"离散化: {problem.discretize}")
#         print(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}")

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
#                 train_x, train_y, pareto_y, pareto_smiles = bayes_opt_molecular_multi(
#                     model, problem, args, init_x, init_y, model_save_dir, device, model_name
#                 )
#                 del model

#                 print(f"\n找到的帕累托前沿大小: {len(pareto_y)}")
#                 print(f"帕累托前沿示例:")
#                 for i, (smiles, y) in enumerate(zip(pareto_smiles[:3], pareto_y[:3])):
#                     print(f"  {i+1}. {smiles}: {y}")
#                 print(f"用时(秒): {time.time() - start_time:.2f}")

#             torch.cuda.empty_cache()

#         os.rename(save_dir, save_dir + "_done")
#         print("多目标优化完成!")
    
#     except Exception as e:
#         print(traceback.format_exc())
#         os.rename(save_dir, save_dir + "_canceled")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="多目标分子优化程序")
#     parser.add_argument("--config", type=str, default="molecular_multi_default", help="配置文件名")
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
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from models import *
import wandb
import numpy as np

from smiles_discrete_problem import SMILESDiscreteProblemMulti

def round_feature_vector(x, problem):
    """根据问题类型对特征向量取整"""
    if problem.discretize:
        return torch.round(x)
    return x

def get_pareto_front(Y):
    """获取帕累托前沿"""
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    
    # 使用botorch的is_non_dominated函数
    pareto_mask = is_non_dominated(Y)
    return Y[pareto_mask], pareto_mask

def compute_hypervolume_indicator(Y, ref_point):
    """计算超体积指标"""
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    
    # 获取帕累托前沿
    pareto_Y, _ = get_pareto_front(Y)
    
    if len(pareto_Y) == 0:
        return 0.0
    
    # 计算超体积
    hv = Hypervolume(ref_point=ref_point)
    return hv.compute(pareto_Y)

def transform_objectives_for_botorch(Y, maximize_flags):
    """
    将目标值转换为botorch期望的格式（全部最大化）
    
    参数:
    ----
    Y: 原始目标值张量，形状为 [n_points, n_objectives]
    maximize_flags: 每个目标是否最大化的布尔列表
    
    返回:
    ----
    转换后的目标值张量（全部变为最大化问题）
    """
    transformed_Y = Y.clone()
    for j, maximize in enumerate(maximize_flags):
        if not maximize:
            # 对于最小化目标，取负数使其变为最大化
            transformed_Y[:, j] = -transformed_Y[:, j]
    return transformed_Y

def compute_reference_hypervolume(problem, ref_point):
    """
    计算数据集中帕累托前沿的参考超体积
    
    参数:
    ----
    problem: SMILESDiscreteProblemMulti实例
    ref_point: 参考点（已经转换为botorch格式，即全部最大化）
    
    返回:
    ----
    参考超体积值
    """
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    
    # 获取数据集中的帕累托前沿
    _, pareto_values, _ = problem.get_pareto_front_from_data()
    
    if len(pareto_values) == 0:
        print("未找到帕累托前沿")
        return 0.0
    
    # 转换为张量
    pareto_Y = torch.tensor(pareto_values, dtype=torch.float64)
    
    # 转换目标值为botorch格式（全部最大化）
    transformed_pareto_Y = transform_objectives_for_botorch(pareto_Y, problem.maximize)
    
    # 计算参考超体积
    hv = Hypervolume(ref_point=ref_point)
    print(f"计算参考超体积，参考点: {ref_point}, 帕累托前沿大小: {len(transformed_pareto_Y)}")
    return hv.compute(transformed_pareto_Y)

def compute_log_hypervolume_difference(current_hv, reference_hv):
    """
    计算Log Hypervolume Difference (LHD)指标
    
    参数:
    ----
    current_hv: 当前超体积
    reference_hv: 参考超体积（通常是数据集帕累托前沿的超体积）
    
    返回:
    ----
    LHD值：log10(reference_hv - current_hv + epsilon)
    """
    # 添加小的epsilon防止取对数时出现负数或零
    epsilon = 1e-10
    print(f"当前超体积: {current_hv}, 参考超体积: {reference_hv}")
    
    difference = reference_hv - current_hv + epsilon
    print(f"超体积差值: {difference}, epsilon: {epsilon}")
    return np.log10(difference)

def bayes_opt_molecular_multi(model, problem, args, init_x, init_y, model_save_dir, device, model_name):
    """
    针对多目标分子优化的贝叶斯优化函数
    
    参数:
    ----
    model: 模型实例
    problem: SMILESDiscreteProblemMulti实例
    args: 配置参数
    init_x: 初始特征向量
    init_y: 初始目标值 (shape: [n_points, n_objectives])
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
    print("初始值形状:", train_y.shape)
    print("初始值示例:", train_y[:3])

    # 记录对应的SMILES
    train_smiles = []
    for i in range(len(train_x)):
        smiles = problem.get_smiles_from_feature(train_x[i].cpu().numpy())
        train_smiles.append(smiles)
    
    # 保存初始分子集
    objective_names = args["objective_names"]
    with open(f"{model_save_dir}/initial_molecules.csv", "w") as f:
        header = "SMILES," + ",".join(objective_names) + "\n"
        f.write(header)
        for smiles, y in zip(train_smiles, train_y):
            values = ",".join([str(val.item()) for val in y])
            f.write(f"{smiles},{values}\n")

    # ========== 统一的参考点设置逻辑 ==========
    # 获取整个数据集的目标值
    all_Y = torch.tensor(problem.y, dtype=torch.float64, device=train_y.device)
    
    print("\n整个数据集的目标值统计:")
    for i, obj_name in enumerate(objective_names):
        print(f"  {obj_name}: min={all_Y[:, i].min():.4f}, max={all_Y[:, i].max():.4f}, "
              f"mean={all_Y[:, i].mean():.4f}, std={all_Y[:, i].std():.4f}")
    
    # 转换目标值为botorch格式（全部最大化）
    transformed_all_Y = transform_objectives_for_botorch(all_Y, args["maximize"])
    
    # 设置参考点（在转换后的空间中，全部都是最大化问题）
    ref_point = torch.zeros(output_dim, dtype=torch.float64, device=train_y.device)
    
    # for i, obj_name in enumerate(objective_names):
    #     # 在转换后的空间中，所有目标都是最大化的
    #     # 参考点应该在转换后的最小值之下
    #     min_val = transformed_all_Y[:, i].min()
    #     range_val = transformed_all_Y[:, i].max() - transformed_all_Y[:, i].min()
    #     ref_point[i] = min_val - 0.1 * range_val
        
    #     if args["maximize"][i]:
    #         print(f"  {obj_name} (最大化): 转换后参考点={ref_point[i]:.4f} < 转换后最小值={min_val:.4f}")
    #     else:
    #         print(f"  {obj_name} (最小化): 转换后参考点={ref_point[i]:.4f} < 转换后最小值={min_val:.4f}")

    # print(f"\n最终参考点（转换后空间）: {ref_point}")

    # 转换目标值为botorch格式（全部最大化）
    transformed_all_Y = transform_objectives_for_botorch(all_Y, args["maximize"])
    
    # 设置参考点（使用与第二段代码相同的逻辑）
    ref_point = torch.zeros(output_dim, dtype=torch.float64, device=train_y.device)
    
    for i, obj_name in enumerate(objective_names):
        if args["maximize"][i]:  # 最大化目标
            # 参考点 = 原始最小值 - 0.1 * |原始最小值|
            original_min_val = all_Y[:, i].min()
            ref_point_original = original_min_val - 0.1 * torch.abs(original_min_val)
            # 转换到botorch空间（最大化目标保持不变）
            ref_point[i] = ref_point_original
            print(f"  {obj_name} (最大化): 原始参考点={ref_point_original:.4f}, 转换后参考点={ref_point[i]:.4f}")
        else:  # 最小化目标
            # 参考点 = 原始最大值 + 0.1 * |原始最大值|
            original_max_val = all_Y[:, i].max()
            ref_point_original = original_max_val + 0.1 * torch.abs(original_max_val)
            # 转换到botorch空间（最小化目标取负数）
            ref_point[i] = -ref_point_original
            print(f"  {obj_name} (最小化): 原始参考点={ref_point_original:.4f}, 转换后参考点={ref_point[i]:.4f}")

    print(f"\n最终参考点（转换后空间）: {ref_point}")
    
    # 计算参考超体积（数据集帕累托前沿的超体积）
    try:
        reference_hv = compute_reference_hypervolume(problem, ref_point)
        print(f"参考超体积（数据集帕累托前沿）: {reference_hv:.6f}")
    except Exception as e:
        print(f"警告: 无法计算参考超体积，将使用默认值: {e}")
        reference_hv = 1.0  # 使用默认值

    # 使用wandb跟踪实验
    use_wandb = args.get("use_wandb", True)
    if use_wandb:
        wandb_project = args.get("wandb_project", "Molecular_Multi_Optimization")
        wandb.init(
            project=wandb_project,
            name=f"{model_name}-{'-'.join(objective_names)}-{args.get('acquisition_function', 'qehvi').upper()}",
            config=args
        )
        
        # 记录参考超体积
        wandb.log({"reference_hypervolume": reference_hv}, step=0)

    # 主优化循环
    for i in range(args["n_BO_iters"]):
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"\n迭代 {i+1}/{args['n_BO_iters']}")

        # 在归一化的特征空间上拟合模型
        model_start = time.time()
        normalized_x = train_x
        model.fit_and_save(normalized_x, train_y, model_save_dir)
        model_end = time.time()
        print(f"模型拟合时间: {model_end - model_start:.2f}秒")
        
        # 构建采集函数
        acq_start = time.time()
        sampler = StochasticSampler(sample_shape=torch.Size([128]))
        
        # 根据配置选择采集函数类型
        acq_type = args.get("acquisition_function", "qehvi").lower()
        
        # 转换训练数据为botorch格式（全部最大化）
        transformed_train_y = transform_objectives_for_botorch(train_y, args["maximize"])
        
        if acq_type == "qehvi":
            # 使用qExpectedHypervolumeImprovement
            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=transformed_train_y
            )
            
            acquisition = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler
            )
            print(f"使用qEHVI采集函数")
            
        elif acq_type == "qnehvi":
            # 使用qNoisyExpectedHypervolumeImprovement
            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=transformed_train_y
            )
            
            acquisition = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                X_baseline=normalized_x,
                sampler=sampler,
                prune_baseline=True
            )
            print(f"使用qNEHVI采集函数")
            
        else:
            raise ValueError(f"不支持的采集函数类型: {acq_type}. 支持的类型: qehvi, qnehvi")
        
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
        # 确保输出维度正确
        if new_y.dim() == 1:
            new_y = new_y.unsqueeze(0)
        
        # 获取对应的SMILES
        new_smiles = problem.get_smiles_from_feature(new_x.cpu().numpy())
        train_smiles.append(new_smiles)
        
        # 扩展训练集
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        
        # 计算当前帕累托前沿（使用转换后的目标值）
        transformed_current_y = transform_objectives_for_botorch(train_y, args["maximize"])
        pareto_y, pareto_mask = get_pareto_front(transformed_current_y)
        pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if pareto_mask[j]]
        
        # 计算超体积指标（使用转换后的目标值）
        hv = compute_hypervolume_indicator(transformed_current_y, ref_point)
        
        # 计算Log Hypervolume Difference (LHD)
        lhd = compute_log_hypervolume_difference(hv, reference_hv)
        
        print(f"新评估的分子: {new_smiles}")
        print(f"新评估的值: {new_y.squeeze()}")
        print(f"当前帕累托前沿大小: {len(pareto_y)}")
        print(f"当前超体积: {hv:.6f}")
        print(f"Log Hypervolume Difference (LHD): {lhd:.6f}")
        
        # 记录到wandb
        if use_wandb:
            log_data = {
                "iteration": i + 1,
                "pareto_front_size": len(pareto_y),
                "hypervolume": hv,
                "log_hypervolume_difference": lhd,
                "acquisition_function": acq_type
            }
            
            # 记录每个目标的新值
            for j, obj_name in enumerate(objective_names):
                log_data[f"new_{obj_name}"] = new_y.squeeze()[j].item()
            
            # 记录帕累托前沿的统计信息（使用原始目标值）
            if len(pareto_mask) > 0:
                original_pareto_y = train_y[pareto_mask]
                for j, obj_name in enumerate(objective_names):
                    log_data[f"pareto_{obj_name}_mean"] = original_pareto_y[:, j].mean().item()
                    log_data[f"pareto_{obj_name}_std"] = original_pareto_y[:, j].std().item()
                    log_data[f"pareto_{obj_name}_min"] = original_pareto_y[:, j].min().item()
                    log_data[f"pareto_{obj_name}_max"] = original_pareto_y[:, j].max().item()
                
            wandb.log(log_data)

    # 保存结果
    if model_save_dir is not None:
        torch.save(train_x.cpu(), f"{model_save_dir}/train_x.pt")
        torch.save(train_y.cpu(), f"{model_save_dir}/train_y.pt")
        
        # 保存所有评估过的分子
        with open(f"{model_save_dir}/evaluated_molecules.csv", "w") as f:
            header = "SMILES," + ",".join(objective_names) + "\n"
            f.write(header)
            for smiles, y in zip(train_smiles, train_y):
                values = ",".join([str(val.item()) for val in y])
                f.write(f"{smiles},{values}\n")
        
        # 保存帕累托前沿（使用原始目标值）
        final_transformed_y = transform_objectives_for_botorch(train_y, args["maximize"])
        final_pareto_y, final_pareto_mask = get_pareto_front(final_transformed_y)
        final_pareto_smiles = [train_smiles[j] for j in range(len(train_smiles)) if final_pareto_mask[j]]
        final_original_pareto_y = train_y[final_pareto_mask]  # 保存原始目标值
        
        with open(f"{model_save_dir}/pareto_front.csv", "w") as f:
            header = "SMILES," + ",".join(objective_names) + "\n"
            f.write(header)
            for smiles, y in zip(final_pareto_smiles, final_original_pareto_y):
                values = ",".join([str(val.item()) for val in y])
                f.write(f"{smiles},{values}\n")
        
        # 保存优化摘要
        final_hv = compute_hypervolume_indicator(final_transformed_y, ref_point)
        final_lhd = compute_log_hypervolume_difference(final_hv, reference_hv)
        
        with open(f"{model_save_dir}/optimization_summary.txt", "w") as f:
            f.write(f"多目标优化摘要\n")
            f.write(f"目标: {', '.join(objective_names)}\n")
            f.write(f"最大化/最小化: {args['maximize']}\n")
            f.write(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}\n")
            f.write(f"总评估次数: {len(train_y)}\n")
            f.write(f"最终帕累托前沿大小: {len(final_original_pareto_y)}\n")
            f.write(f"最终超体积: {final_hv:.6f}\n")
            f.write(f"参考超体积: {reference_hv:.6f}\n")
            f.write(f"最终Log Hypervolume Difference (LHD): {final_lhd:.6f}\n")
            f.write(f"参考点（转换后空间）: {ref_point.tolist()}\n")
        
        # 保存LHD历史记录
        lhd_history = []
        for step in range(len(train_y) - len(init_y) + 1):  # 包括初始状态
            if step == 0:
                # 初始状态
                current_transformed_y = transform_objectives_for_botorch(init_y, args["maximize"])
                current_hv_step = compute_hypervolume_indicator(current_transformed_y, ref_point)
            else:
                # 第step次迭代后的状态
                current_train_y = torch.cat([init_y, train_y[len(init_y):len(init_y)+step]])
                current_transformed_y = transform_objectives_for_botorch(current_train_y, args["maximize"])
                current_hv_step = compute_hypervolume_indicator(current_transformed_y, ref_point)
            
            lhd_step = compute_log_hypervolume_difference(current_hv_step, reference_hv)
            lhd_history.append({
                "iteration": step,
                "hypervolume": current_hv_step,
                "lhd": lhd_step
            })
        
        # 保存LHD历史为CSV
        import pandas as pd
        lhd_df = pd.DataFrame(lhd_history)
        lhd_df.to_csv(f"{model_save_dir}/lhd_history.csv", index=False)

    if use_wandb:
        # 记录最终的LHD值
        wandb.log({"final_lhd": final_lhd})
        wandb.finish()

    return train_x, train_y, final_original_pareto_y, final_pareto_smiles


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
    objective_names = args["objective_names"]
    dataset_name = args["dataset_name"]
    
    objectives_str = "-".join(objective_names)
    if cl_args.name:
        save_dir = f"{save_dir}_{cl_args.name}_{dataset_name}_{objectives_str}"
    else:
        save_dir = f"{save_dir}_{cl_args.config}_{dataset_name}_{objectives_str}"
    
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
        device = torch.device("cpu")
        torch.set_default_dtype(torch.float64)

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

        # 创建多目标优化问题
        print(f"创建多目标SMILES离散优化问题...")
        problem = SMILESDiscreteProblemMulti(
            smiles_data_path=dataset_path,
            objective_names=objective_names,
            feature_type=args.get("feature_type", "morgan"),
            fp_radius=args.get("fp_radius", 2),
            fp_bits=args.get("fp_bits", 60),
            discretize=args.get("discretize", True),
            n_bins=args.get("n_bins", 2),
            maximize=args.get("maximize", [True] * len(objective_names)),
            seed=int(args["seed"])
        )
        
        # 保存问题定义
        problem.save(os.path.join(save_dir, "problem.pkl"))
        
        # 获取维度
        input_dim = problem.dim
        output_dim = len(objective_names)
        
        print(f"问题维度: {input_dim}")
        print(f"目标数量: {output_dim}")
        print(f"目标名称: {objective_names}")
        print(f"最大化/最小化: {args['maximize']}")
        print(f"特征类型: {problem.feature_type}")
        print(f"离散化: {problem.discretize}")
        print(f"采集函数: {args.get('acquisition_function', 'qehvi').upper()}")

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
                train_x, train_y, pareto_y, pareto_smiles = bayes_opt_molecular_multi(
                    model, problem, args, init_x, init_y, model_save_dir, device, model_name
                )
                del model

                print(f"\n找到的帕累托前沿大小: {len(pareto_y)}")
                print(f"帕累托前沿示例:")
                for i, (smiles, y) in enumerate(zip(pareto_smiles[:3], pareto_y[:3])):
                    print(f"  {i+1}. {smiles}: {y}")
                print(f"用时(秒): {time.time() - start_time:.2f}")

            torch.cuda.empty_cache()

        os.rename(save_dir, save_dir + "_done")
        print("多目标优化完成!")
    
    except Exception as e:
        print(traceback.format_exc())
        os.rename(save_dir, save_dir + "_canceled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多目标分子优化程序")
    parser.add_argument("--config", type=str, default="molecular_multi_default", help="配置文件名")
    parser.add_argument("--bg", action="store_true", help="是否在后台运行")
    parser.add_argument("-n", "--name", type=str, help="实验名称(可选)")
    cl_args = parser.parse_args()

    main(cl_args)


