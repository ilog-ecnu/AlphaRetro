from multiprocessing import Pool
from rdkit import Chem
import numpy as np
import time
import yaml
import sys
import torch
from utils import cal_similarity_with_FP, get_kth_submole_from_smile, generate_scaffold, route_save_condition,get_scaff_list, read_txt
from model import GeneticAlgorithm, GA_branch
from single_step.t5.client import run_client
from single_step.rxnfp.client import client_rx


class RetroSynthesis():
    def __init__(self, problem_name='DemoA', config_path=None, **kwargs):
        self.problem_name = problem_name
        self.avg_calls = []
        self.list_split = lambda lst: (lst[:len(lst)//2], lst[len(lst)//2:])  # 定义一个切分list的函数

        if config_path is not None:
            with open(config_path, 'r') as f:
                configs= yaml.safe_load(f)
                self.config = configs[self.problem_name]
        else:
            raise ValueError('config_path is None')

    def cal_score(self, react_smile):
        react_smiles = react_smile.split('.')
        if len(react_smiles) > 2:  # 如果最后一个分子是由超过2个分子组成的，score直接返回一个较小的值
            return -10
        else:
            scaff_list = get_scaff_list(self.config['building_block_dataset'])  # 获取原料库所有分子的骨架--->smile list
            output_score = []
            for smile in react_smiles:
                compare_scaff_list = []
                for line in scaff_list:
                    smile_score = cal_similarity_with_FP(generate_scaffold(smile), line)
                    compare_scaff_list.append(smile_score)
                # output_score.append(max(compare_scaff_list))
                output_score.append(np.mean(compare_scaff_list))
            return np.mean(output_score)
        
    def cal_score_smile(self, react_smile):
        react_smiles = react_smile.split('.')
        if len(react_smiles) > 2:  # 如果最后一个分子是由超过2个分子组成的，score直接返回一个较小的值
            return -10
        else:
            smile_list = read_txt(self.config['building_block_dataset'])  # 获取原料库所有分子的骨架--->smile list
            output_score = []
            for smile in react_smiles:
                compare_scaff_list = []
                for line in smile_list:
                    smile_score = cal_similarity_with_FP(smile, line)
                    compare_scaff_list.append(smile_score)
                # output_score.append(max(compare_scaff_list))
                output_score.append(np.mean(compare_scaff_list))
            return np.mean(output_score)
    
    def evaluate_indivisual(self, iter_num, indis, GPU_ID):
        react_smi = [self.config['target_smi']]
        react_chain = ['<RX_' + client_rx({'input_text':self.config['target_smi']}) + '>' + self.config['target_smi']]  # 自动
        react_chain_index = []
        
        individual, branch = self.list_split(indis)
        for j, indi in enumerate(individual):  # 遍历个体的每个节点 ，indi为0-1之间的值，需要转换成react
            if j == 0:
                target_smi = '<RX_' + client_rx({'input_text':self.config['target_smi']}) + '>' + self.config['target_smi']
                data = {'input_text': target_smi,'gpu_id': GPU_ID, 'beam_size': self.config['beam_size']}
                result = run_client(data)
                smi_nodes_sorted, score = result['result'], result['score']
                current_smile = smi_nodes_sorted[indi]
                
                if Chem.MolFromSmiles(current_smile)== None or len(current_smile.split('.')) > 2:  # 如果当前节点无效，结束当前个体的评估，剪枝(大于2分支任务需修改此处)
                    return -10

                main_node = get_kth_submole_from_smile(current_smile, branch[j])
                rx = '<RX_' + client_rx({'input_text':main_node}) + '>'  # rx
                next_node = rx + main_node
                react_smi.append(current_smile)  # 不带有反应类型的反应路线
                react_chain.append(next_node)  # 带有反应类型的反应路线
                react_chain_index.append(indi)
            else:
                data = {'input_text': next_node,'gpu_id': GPU_ID, 'beam_size': self.config['beam_size']}
                result = run_client(data)
                smi_nodes_sorted, score = result['result'], result['score']
                current_smile = smi_nodes_sorted[indi]  # 单步结果选择top indi
                
                if Chem.MolFromSmiles(current_smile)== None  or len(current_smile.split('.')) > 2:  # 如果当前节点无效，结束当前个体的评估，剪枝(大于2分支任务需修改此处)
                    return -10 
                
                main_node = get_kth_submole_from_smile(current_smile, branch[j])
                rx = '<RX_' + client_rx({'input_text':main_node}) + '>'  # rx
                next_node = rx + main_node
                react_smi.append(current_smile)  # 不带有反应类型的反应路线
                react_chain.append(next_node)  # 带有反应类型的反应路线
            
                if route_save_condition(main_node, self.config['building_block_dataset']):
                    print('call_list---: {}'.format(' --> '.join(react_chain)))
                react_chain_index.append(indi)

        reaction_smi = ".".join(react_smi)
        try:
            m = Chem.MolFromSmiles(reaction_smi)
            smi_score = self.cal_score(react_smi[-1])  # react_smi[-1]为每个反应路径的最后一个节点
        except:
            smi_score = 0
            print('Chem.MolFromSmiles返回空, 导致sim_score_ground_truth异常')
        
        self.avg_calls.append(sys.getrefcount(run_client))
        print('iter: {}, sum_calls: {}'.format(iter_num, int(np.sum(self.avg_calls))))

        return smi_score

    def _evaluate(self, iter_num, i, indis):  # indis是个体
        available_GPU = [0,1,2,3]  # [0,1,2,3]  [0,1,2]
        GPU_ID = i % len(available_GPU)
        return self.evaluate_indivisual(iter_num, indis, available_GPU[GPU_ID])
    
def _f_x(score_cls_list, eval_method):
    observe = []
    for score_cls in score_cls_list:
        observe.append(score_cls.get())
    if eval_method == 1:
        return (-1) * np.array(observe)
    elif eval_method == 2:
        score = np.exp(np.array(observe) * 2)  # 扩大与目标值间的距离
    elif eval_method == 3:
        score = np.random.choice(observe, size=len(observe), replace=True, p=np.array(observe)/sum(observe))
    return (-1) * np.array(score)
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    num_worker = 1  # 线程池数量
    fx = 1  # 2,3  # fx评估函数
    epoch = 100  # 迭代次数
    pop_size = 20  # 通常和线程池数量num_worker成倍数关系
    k_cross = 2  # k叉

    problem = RetroSynthesis(problem_name='DemoA', config_path='config.yaml')

    main_prob = GeneticAlgorithm(population_size=pop_size, genome_length=problem.config['step_len'], beam_size=problem.config['beam_size'], crossover_rate=0.7, mutation_rate=0.1)
    pops = main_prob.init_population()  # 初始化population

    branch_prob = GA_branch(population_size=pop_size, genome_length=problem.config['step_len'], beam_size=k_cross, crossover_rate=0.7, mutation_rate=0.1)
    branchs = branch_prob.init_population()  # 初始化k叉

    avg_iter_time = []
    for iter_num in range(epoch):
        start_time = time.time()
        offs_pops = main_prob.create_new_generation(pops)
        offs_branch = branch_prob.create_new_generation_flip(branchs)  # 只用 0-1 互变
        
        # 合并当前 pop and off_pop
        pops_and_off_pops = np.vstack([pops, offs_pops])
        branch_and_off_branch = np.vstack([branchs, offs_branch])
        Xs = np.hstack([pops_and_off_pops, branch_and_off_branch])

        score_cls_list = []  # 每个个体的分数，转换成最小值优化

        # # 多线程调用
        pool = Pool(num_worker)
        for i, indis in enumerate(Xs):  # 遍历种群中每个个体
            score_cls_list.append(pool.apply_async(func=problem._evaluate, args=(iter_num, i, indis)))
        pool.close()
        pool.join()
        ys = _f_x(score_cls_list, fx)

        # 单线程dubug
        # for i, indis in enumerate(Xs):  # 遍历种群中每个个体
        #     score = problem._evaluate(iter_num, i, indis)
        #     score_cls_list.append(score)
        # ys = (-1) * np.array(score_cls_list)

        # select the best pop_size individuals
        index = np.argsort(ys)  # 升序找下标

        pops = pops_and_off_pops[index[:pop_size], :]
        branchs = branch_and_off_branch[index[:pop_size], :]

        elapsed_time = int(time.time()-start_time)
        avg_iter_time.append(elapsed_time)
        print('iter: {}, loss: {}, elapsed_time: {}s, avg_iter_time: {}s, total_time: {}s'.format(iter_num, round(min(ys),4), elapsed_time, int(np.mean(avg_iter_time)), sum(avg_iter_time)))
