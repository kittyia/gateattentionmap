import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from policy_manager import PolicyManager

def entropy_ind(x):
    if x==0:
        return 0
    else:
        return -x*np.log(x)

def fairness(data):
    p1 = data.iloc[-1,:]['fg1']
    p2 = data.iloc[-1,:]['fg2']
    if p1+p2 == 0:
        return 0
    else:
        return np.power(p1-p2,2)/np.power(p1+p2,2)

'''
def fairness(data):
        
        begin_fair = 0
        if data.iloc[0,:]['fg1'] == 0:
            begin_fair = 0
        else:
            begin_fair = data.iloc[0,:]['fg2'] / data.iloc[0,:]['fg1']
        
        end_fair1 = 0
        end_fair2 = 0
        if data.iloc[-1,:]['fg1'] == 0:
            end_fair1 = 0
        else:
            end_fair1 = data.iloc[-1,:]['fg2'] / data.iloc[-1,:]['fg1']

        if data.iloc[-1,:]['fg2'] == 0:
            end_fair2 = 0
        else:
            end_fair2 = data.iloc[-1,:]['fg1'] / data.iloc[-1,:]['fg2']
        return np.power(end_fair1 - 1,2)+np.power(end_fair2 - 1,2)
'''
#

def begin_sensitive(data):
    return data.iloc[0,0]+data.iloc[0,1]




def prep(data):
    out_df = data['prep'].apply(entropy_ind)
    out = out_df.sum()
    return out

# state
def average_state(data):
    return data['fg1'].mean(),data['fg2'].mean(), data['fg3'].mean()

def var_state(data):
    return data['fg1'].var(),data['fg2'].var(), data['fg3'].var()

# pro
def average_pro(data):
    return data['prg1'].mean(),data['prg2'].mean(), data['prg3'].mean()


def var_pro(data):
    return data['prg1'].var(),data['prg2'].var(), data['prg3'].var()

# post
def average_post(data):
    return data['pog1'].mean(),data['pog2'].mean(), data['pog3'].mean()

def var_post(data):
    return data['pog1'].var(),data['pog2'].var(), data['pog3'].var()


def get_collections(inpath,outfile):
    print('--get collections')
    policys_idx = list()
    files_idx =list()

    for root, dirs, files in os.walk('outputs'):
        for file in files:
            print(root,file)
            nums = file.split('_')
            # print(nums[0],nums[1][:-4])
            policys_idx.append(nums[1][:-4])
            files_idx.append(nums[0])

    policys_idx = list(set(policys_idx))
    policys_idx.sort()
    print(policys_idx)



    files_idx = list(set(files_idx))
    files_idx.sort()
    print(files_idx)

    out_df = pd.DataFrame()
    idx = 0
    for root, dirs, files in os.walk(inpath):
        for file in files:
            # print(root,file)
            nums = file.split('_')

            now_data = pd.read_csv(os.path.join(root,file))
            # print(now_data)
            new_row = dict()
            new_row['prompt']= nums[0]
            new_row['policy']=policys_idx.index(nums[1][:-4])
            # new_row['policy']=nums[1][:-4]
            new_row['sensitive']=begin_sensitive(now_data)

            new_row['ave_state1'], new_row['ave_state2'], new_row['ave_state3']=average_state(now_data)
            new_row['var_state1'], new_row['var_state2'], new_row['var_state3']=var_state(now_data)

            new_row['ave_pr1'], new_row['ave_pr2'], new_row['ave_pr3']=average_pro(now_data)
            new_row['var_pr1'], new_row['var_pr2'], new_row['var_pr3']=var_pro(now_data)
            
            new_row['ave_po1'], new_row['ave_po2'], new_row['ave_po3']=average_post(now_data) 
            new_row['var_po1'], new_row['var_po2'], new_row['var_po3']=var_post(now_data)

            new_row['preplexity']=prep(now_data)
            new_row['fairness'] = fairness(now_data)


            new_row = pd.DataFrame(new_row,index=[idx])
            idx += 1

            out_df = pd.concat([out_df, new_row], ignore_index=True)

    out_df.to_csv(outfile)
    return out_df
    
def stat(data, outfile):
    
    data=data[data['sensitive'] != 0]
    data=data[(data != np.inf).any(axis=1)]

    print(len(data))
    print(data)
    # 查看每种policy的各种值
    grouped_mean = data.groupby('policy', as_index=True).mean()
    print(grouped_mean)
    grouped_mean.to_csv(outfile)

    grouped_var = data.groupby('policy', as_index=True).var()
    grouped_var.to_csv('var.csv')
    return grouped_mean


def compute_stats(df, column):
    stats = df.groupby('policy')[column].agg(['mean', 'var', 
                                             lambda x: x.quantile(0.25),
                                             lambda x: x.quantile(0.5),
                                             lambda x: x.quantile(0.75)])
    stats.columns = ['mean', 'variance', 'Q1_25%', 'Q2_50%', 'Q3_75%']  # 重命名列
    return stats

# 选择需要比较的列
def see_column_stat(data,target_column):
    # 1. 按 policy 分组，计算均值和标准差
    grouped_stats = data.groupby('policy')[target_column].agg(['mean', 'std'])

    # 2. 提取数据
    policies = grouped_stats.index  # 分组的名字（如 A、B、C）
    means = grouped_stats['mean']   # 每组的均值
    stds = grouped_stats['std']     # 每组的标准差

    # 3. 绘制柱状图，添加误差条
    plt.figure(figsize=(8, 6))
    plt.bar(policies, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7)

    # 4. 添加图表标签和标题
    plt.xlabel('Policy')
    plt.ylabel('Fairness Mean with Std')
    plt.title('Fairness Mean and Standard Deviation by Policy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 显示图表
    plt.tight_layout()
    plt.show()

if __name__ =='__main__':
    raw_path = 'outputs'
    collect_file_name = 'collection.csv'
    stat_file_name = 'stat.csv'

    
    if not os.path.exists(collect_file_name):
        data = get_collections(raw_path,collect_file_name)
    else:
        data = pd.read_csv(collect_file_name,index_col=0)
    

    # data = get_collections(raw_path,collect_file_name)
    policy_processor = PolicyManager()
    policys = policy_processor.total()
    policys.sort()
    for i in range(len(policys)):
        print('#',i,'\t',policys[i])

    see_column_stat(data,'fairness')
    
    # stat(data,stat_file_name)
    
   
    