import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def entropy_ind(x):
    if x==0:
        return 0
    else:
        # return -x*np.log(x)
        return -np.log(x)

def fairness(data):
    p1 = data.iloc[-1,:]['fg1']
    p2 = data.iloc[-1,:]['fg2']
    if p1+p2 == 0:
        return 0
    else:
        return np.power(p1-p2,2)/np.power(p1+p2,2)

def fairness_local(data):
    n1 = data.iloc[-1,:]['fg1']
    n2 = data.iloc[-1,:]['fg2']
    if n1+n2 == 0:
        return 0
    else:
        p1 = n1/(n1+n2)
        p2= n2/(n1+n2)
        return np.fabs(p1-p2)

def fairness_direct(data):
    n1 = data.iloc[-1,:]['fg1']
    n2 = data.iloc[-1,:]['fg2']
    if n1+n2 == 0:
        return 0
    else:
        p1 = n1/(n1+n2)
        p2= n2/(n1+n2)
        return p1-p2

def fairness_global(data):
    n1 = data.iloc[-1,:]['fg1']
    n2 = data.iloc[-1,:]['fg2']
    n3 = data.iloc[-1,:]['fg3']
    if n1+n2 == 0:
        return 0
    else:
        p1 = n1/(n1+n2+n3)
        p2 = n2/(n1+n2+n3)
        return np.fabs(p1-p2)

def begin_sensitive(data):
    return data.iloc[0,0]+data.iloc[0,1]

def begin_bias(data):
    return np.fabs(data.iloc[0,0]-data.iloc[0,1])


def preplexicy(data):
    out_df = data['prep'].apply(entropy_ind)
    out = np.power(2,out_df.sum()/len(data))
    return out

def move(data):
    out = 0
    for index,row in data.iterrows():
        if row['action']==1:
            out += np.fabs(row['prg1']-row['pog1'])
            out += np.fabs(row['prg2']-row['pog2'])
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


# transition probability
def average_yt_cos(data):
    out = list()
    init = [data.loc[0,'fg1'],data.loc[0,'fg2']]
    for index,row in data.iterrows():
        # Y_t group count 
        vector1=[row['fg1']-init[0]+1,row['fg2']-init[1]+1]
        # logit distritbution among group ; currently
        vector2 =  [row['prg1']+1,row['prg2']+1]
        
        dot_product = np.dot(vector1, vector2)  # 向量点积
        if dot_product !=0:
            norm_vector1 = np.linalg.norm(vector1)  # 向量1的模
            norm_vector2 = np.linalg.norm(vector2)  # 向量2的模
            similarity = dot_product / (norm_vector1 * norm_vector2)
        else:
            similarity = 1
        out.append(similarity)
    return sum(out) / len(out)

def average_ytend_cos(data):
    out = list()
    len_prompt = data.loc[0,'fg1']+data.loc[0,'fg2']+data.loc[0,'fg3']
    rate = len(data)*len_prompt
    init = [data.loc[0,'fg1'],data.loc[0,'fg2']]
    purpose = [data.loc[0,'fg1']*rate, data.loc[0,'fg2']*rate]
    for index,row in data.iterrows():
        vector1 =  [
            purpose[0]-(row['prg1']-init[0])+1,
            purpose[1]-(row['prg2']-init[1])+1
            ]
        vector2=[row['prg1'],row['prg2']]
        dot_product = np.dot(vector1, vector2)  # 向量点积
        if dot_product !=0:
            norm_vector1 = np.linalg.norm(vector1)  # 向量1的模
            norm_vector2 = np.linalg.norm(vector2)  # 向量2的模
            similarity = dot_product / (norm_vector1 * norm_vector2)
        else:
            similarity = 1
        out.append(similarity)
    return sum(out) / len(out)

def average_xyt_cos(data):
    out = list()
    for index,row in data.iterrows():
        vector1=[row['fg1']+1,row['fg2']+1] 
        vector2 =  [row['prg1']+1,row['prg2']+1]
        dot_product = np.dot(vector1, vector2)  # 向量点积
        if dot_product !=0:
            norm_vector1 = np.linalg.norm(vector1)  # 向量1的模
            norm_vector2 = np.linalg.norm(vector2)  # 向量2的模
            similarity = dot_product / (norm_vector1 * norm_vector2)
        else:
            similarity = 1
        out.append(similarity)
    return sum(out) / len(out)

def get_collections(save_path):
    print('--get collections')
    policy_path = 'policy_ex'

    out_df = pd.DataFrame()

    for root, dirs, files in os.walk(policy_path):
        for file in files:
            # print(file)    
            policy_name = file[:-4]
            policy_process_path = os.path.join('outputs',f'process_{policy_name}')
            print(policy_process_path)
            #
            idx = 0
            for root1, dirs1, files1 in os.walk(policy_process_path):
                for piece_file in files1:
                    print(piece_file)
                    nums = piece_file.split('_')
                    # print(nums[0],nums[1][:-4])
                    
                    now_data = pd.read_csv(os.path.join(root1,piece_file))
                    # print(now_data)
                    new_row = dict()
                    new_row['prompt_idx']= nums[0]
                    
                    new_row['policy']=policy_name

                    new_row['prompt_distribution']= '{}_{}'.format(now_data.iloc[0,0],now_data.iloc[0,1])
                    new_row['init_male']=now_data.iloc[0,0]
                    new_row['init_female']=now_data.iloc[0,1]

                    new_row['sensitive']=begin_sensitive(now_data)
                    new_row['bias']=begin_bias(now_data)

                    # new_row['init_sensitive'] = begin_sensitive(now_data)
                    # new_row['policy']=nums[1][:-4]

                    new_row['init_group1']=now_data.iloc[0,0]
                    new_row['init_group2']=now_data.iloc[0,1]
                    new_row['init_group3']=now_data.iloc[0,2]

                    new_row['end_group1']=now_data.iloc[-1,0]
                    new_row['end_group2']=now_data.iloc[-1,1]
                    new_row['end_group3']=now_data.iloc[-1,2]

                    

                    new_row['ave_state1'], new_row['ave_state2'], new_row['ave_state3']=average_state(now_data)
                    new_row['var_state1'], new_row['var_state2'], new_row['var_state3']=var_state(now_data)

                    new_row['ave_pr1'], new_row['ave_pr2'], new_row['ave_pr3']=average_pro(now_data)
                    new_row['var_pr1'], new_row['var_pr2'], new_row['var_pr3']=var_pro(now_data)
                    
                    new_row['ave_po1'], new_row['ave_po2'], new_row['ave_po3']=average_post(now_data) 
                    new_row['var_po1'], new_row['var_po2'], new_row['var_po3']=var_post(now_data)

                    new_row['preplexity']=preplexicy(now_data)
                    new_row['move']=move(now_data)

                    new_row['fairness_local'] = fairness_local(now_data)
                    new_row['fairness_global'] = fairness_global(now_data)
                    new_row['fairness_direct'] = fairness_direct(now_data)

                    new_row['average_xyt_cos'] = average_xyt_cos(now_data)
                    new_row['average_yt_cos'] = average_yt_cos(now_data)
                    new_row['average_ytend_cos'] = average_ytend_cos(now_data)

                    new_row = pd.DataFrame(new_row,index=[idx])
                    idx += 1

                    out_df = pd.concat([out_df, new_row], ignore_index=True)

    # cluster
    #plt.scatter(out_df['init_male'],out_df['init_female'])
    # plt.savefig(os.path.join(save_path,'1-prompts distritbuion.png'))
    # plt.close()
    out_df.to_csv(os.path.join(save_path,'1-collection.csv'),index=False)
    return out_df

def see_prompts_distribution(data,save_path):
    # plt.hist2d(data['init_male'],data['init_female'])
    result = pd.crosstab(index=data['init_male'], columns=data['init_female'])
    result.to_csv(os.path.join(save_path,'1-prompts distribution.csv'))
    # 打印结果表
    print(result)
    using_data = data[data['sensitive']<10]
    x,y =using_data['init_male'], using_data['init_female']
    # 绘制 KDE 密度图
    sns.kdeplot(x=x, y=y, cmap='cividis', fill=True, thresh=0, levels=100)

    # 添加等高线
    sns.kdeplot(x=x, y=y, cmap='cool', levels=10, linewidths=1.5)
    plt.savefig(os.path.join(save_path,'1-prompts distritbuion.png'))
    plt.close()
    

def see_column_stat(save_path,data,target_column):
    # 1. 按 policy 分组，计算均值和标准差
    prompt_distritbuion_list = list(set(data['prompt_distribution'].tolist()))
    prompt_distritbuion_list.sort()
    
    for prompt_f in prompt_distritbuion_list:

        using_data = data[data['prompt_distribution']==prompt_f]
        # print(using_data)
        # using_data = using_data[using_data['policy'].isin(policy_selection)]
        print(prompt_f)

        custom_order = ['none']
        custom_order += [f"pd{i}rho{j}" for i in range(0,7) for j in [0,2,4,8]]
        print(custom_order)

        print(f' using collect {len(using_data)/len(custom_order)} from {len(data)/len(custom_order)}')
        grouped_stats = using_data.groupby('policy')[target_column]
        
        if len(grouped_stats)>1:
            plt.figure(figsize=(18, 6))
            
            sns.boxplot(x='policy', y=target_column, data=using_data,order=custom_order)

            # 自定义调色板

            # 添加标题和标签
            plt.title(f'Boxplot of {target_column} by Policy')
            plt.xlabel('Policy')
            plt.ylabel(target_column)

        else:
            
            # 4. 添加图表标签和标题
            # 2. 提取数据
            policies = grouped_stats.index  # 分组的名字（如 A、B、C）
            means = grouped_stats['mean']   # 每组的均值
            # stds = grouped_stats['std']     # 每组的标准差
            
            print(means)
            plt.bar(policies, means, capsize=5, color='skyblue', alpha=0.7)

        plt.xlabel('Policy')
        plt.ylabel(f'{target_column} Mean with Std')
        plt.title(f'{target_column}_{prompt_f}_no{len(using_data)/len(custom_order)} Mean and Standard Deviation by Policy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 5. 显示图表
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,f'collect_{target_column}_{prompt_f}.png'))

def analysis_transition_probability(data):
    out_df = pd.DataFrame()
    using_data = data[data['policy']=='none']
    for index,row in using_data.iterrows():
        print(index, row)


def evaluate_output(out_path,input_data):
    idx_list = ['preplexity','move',
                'fairness_local','fairness_global','fairness_direct',
                'average_xyt_cos','average_ytend_cos','average_yt_cos']
    for idx in idx_list:
        see_column_stat(out_path,input_data,idx)

if __name__ =='__main__':
    evaluation_path = 'evaluation'
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    
    df = get_collections(evaluation_path)
    df = pd.read_csv(os.path.join(evaluation_path,'1-collection.csv'))
    
    # see_prompts_distribution(df,evaluation_path)
    
    print(list(set(df['policy'].tolist())))
    # print(df)
    evaluate_output(evaluation_path,df)
    
    
   
    