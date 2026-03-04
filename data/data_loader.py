import pandas as pd
from tqdm import tqdm
import numpy as np
import os

def mle_gbm(prices, dt=1/252):
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    
    # 波动率的极大似然估计
    sigma_hat = np.sqrt(np.sum((log_returns - np.mean(log_returns))**2) / (n * dt))
    
    # 漂移率的极大似然估计
    mu_hat = (np.mean(log_returns) / dt) + 0.5 * sigma_hat**2
    
    return mu_hat, sigma_hat

def calculate_parements_stock(data_path, result_path=None, save_csv=False):

    # 如果不需要计算，直接加载参数文件
    if not save_csv:
        print(f"直接加载参数文件: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"参数文件不存在: {data_path}")
        results_df = pd.read_csv(data_path, encoding="utf-8")
        # 转换为字典列表格式
        results = results_df.to_dict('records')
        return results

    # 需要计算参数：从原始数据计算
    print("开始计算参数...")
    # 加载原始数据
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path, encoding="utf-8")
    elif data_path.endswith('.xlsx') or data_path.endswith('.xlx'):
        df = pd.read_excel(data_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")

    # 按股票分组
    grouped = df.groupby('id')
    # 存储结果
    results = []

    # 主循环：对每个股票运行 GA
    for stock_id, group in tqdm(grouped):
        print(f"\n处理股票: {stock_id}")
        # print(group)
        # 提取真实路径
        real_path = group[-252:]['spj'].values
        mu,sigma = mle_gbm(real_path)
        results.append({
            'stock_id': stock_id,
            'mu': mu,
            'sigma': sigma,
        })
    results_df = pd.DataFrame(results)
    if result_path:
        results_df.to_csv(result_path, index=False)
        print(f"\n所有股票参数优化完成，结果已保存至 '{result_path}'")

    return results


