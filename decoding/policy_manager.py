import configparser
import os
import pandas as pd
import numpy as np

from settings_manager import *


class csPolicy:
    def __init__(self):
        self.name = 'cs_debias'

        self.p_male = 0.5
        self.p_female = 0.5
        
        self.support_time = list()
        
        self.rho = 0


    # current desire fairness bias;
    # i 
    
    def show(self):
        print('--policy--')
        print(' -info')
        print(f'  name: {self.name}')
        print(' -reference')
        print(f'  male: {self.p_male}')
        print(f'  female: {self.p_female}')
        print(' -tolerance')
        print(f'  rho: {self.rho}')
        print(' -support_time')
        print(f'  time: {self.support_time}')

    @staticmethod
    def _policy_str2list(policycode:str):
        binary_representation = bin(int(policycode))[2:]

        positions = [idx for idx, bit in enumerate(reversed(binary_representation)) if bit == '1']
        return positions

    @staticmethod
    def _policy_list2str(policylist:list):
        result = sum(2 ** pos for pos in policylist)
        return str(result)

    def save(self,save_path = 'policy_ex'):
        """
        保存列表到 INI 文件
        :param section: 配置段名
        :param key: 键名
        :param values: 要保存的列表
        :param filename: INI 文件名
        """
        save_file = os.path.join(save_path,f'{self.name}.ini')
        
        config = configparser.ConfigParser()

        #
        config['info'] = {}
        config['info']['name'] = self.name
        #
        config['reference'] = {}
        config['reference']['male'] = str(self.p_male)
        config['reference']['female'] = str(self.p_female)
        # 
        config['tolerance'] = {}
        config['tolerance']['rho'] = str(self.rho)


        config['support_time'] = {}
        config['support_time']['time'] = self._policy_list2str(self.support_time)
        
        with open(save_file, 'w') as configfile:
            config.write(configfile)

    #
    
    def read(self,read_file):


        config = configparser.ConfigParser()
        config.read(read_file)
        
        print(config)
        #
        self.name = config['info']['name']
        #
        self.p_male = float(config['reference']['male'])
        self.p_female = float(config['reference']['female'])
        # 
        self.rho = int(config['tolerance']['rho'])

        self.support_time = self._policy_str2list(config['support_time']['time'])



def generate_policy(args):
    test = csPolicy()
    if not os.path.exists(args.policy_ex_path):
        os.mkdir(args.policy_ex_path)
    # none
    test.name = f'none'
    test.save(args.policy_ex_path)
    '''
    # all
    for j in [0,2,4,8]:
        test.name = f'allrho{j}'
        test.support_time = [t 
                for t in range(0,args.max_new_tokens)] 
        test.rho = j
        # test.show()
        test.save(args.policy_ex_path)
    '''
    # periods i  & tolerance j
    exp_bound = max(1,int(np.log2(args.max_new_tokens)))
    for i in range(0,exp_bound):
        for j in [0,2,4,8]:
            test.name = f'pd{i}rho{j}'
            test.support_time = [t 
                    for t in range(0,args.max_new_tokens)
                    if not t % np.power(2,i)] 
            test.rho = j
            # test.show()
            test.save(args.policy_ex_path)

if __name__ == '__main__':
    args = parse_args()
    generate_policy(args)
    # implimentable