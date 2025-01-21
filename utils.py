import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import subprocess
import json

def normalization(X, var):
    """
    In this Python code, X is a numpy array representing the input matrix, and var is a numpy array containing the lower and upper bounds. 
    The function calculates the normalized matrix Norm_X using element-wise operations. 
    The np.tile() function is used to replicate the lower and upper bounds to match the dimensions of X before performing the normalization.
    """
    L = X.shape[0]
    Lower = var[0, :]
    Upper = var[1, :]
    Norm_X = (X - np.tile(Lower, (L, 1))) / np.tile(Upper - Lower, (L, 1))
    return Norm_X

def cal_similarity_with_FP(source_seq, target_seq):
    # with suppress_stdout_stderr():
	try:
		mol1 = Chem.MolFromSmiles(source_seq)  # 读取单个smiles字符串
		mol2 = Chem.MolFromSmiles(target_seq)
		mol1_fp = AllChem.GetMorganFingerprint(mol1, 2)  # 计算摩根指纹，radius=2 代表原子环境半径为2。 摩根型指纹是一种圆形指纹。每个原子的环境和连通性被分析到给定的半径，并且每种可能性都被编码。通常使用散列算法将很多可能性压缩到预定长度，例如1024。因此，圆形指纹是原子类型和分子连通性的系统探索
		mol2_fp = AllChem.GetMorganFingerprint(mol2, 2)
		score = DataStructs.DiceSimilarity(mol1_fp, mol2_fp)  # 比较分子之间的相似性
		return score
	except:
		return 0

def get_kth_submole_from_smile(tmpseqs, k):
    """选择第k个smile表达式输出"""
    seq_list = tmpseqs.split(".")

    if len(seq_list) == 1:
        main_smi = tmpseqs
    else:
        main_smi = seq_list[k]
    return main_smi

def cano(smiles):
    """标准化判定"""
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi

def generate_scaffold(smiles):
    """
    生成骨架smiles
    """
    mol = Chem.MolFromSmiles(smiles)
    # 生成骨架结构
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # 从骨架结构生成SMILES
    scaffold_smiles = Chem.MolToSmiles(scaffold)

    return scaffold_smiles

def get_scaff_list(database_path):
    """获取原料库中所有原料的骨架list"""
    scaff_list = []
    database = read_txt(database_path)
    for line in database:
        scaff = generate_scaffold(line)
        scaff_list.append(scaff)
    return scaff_list

def save_react_chain(content, save_path):    
	with open(save_path, "a+") as f:
		f.write(content + "\n")

def read_txt(path):
    with open(path, "r", encoding='utf-8') as f:  # 打开文件
        data = f.readlines()  # 读取文件
    return data

def write_txt(content, path):
    with open(path, "a", encoding='utf-8') as f:  # 打开文件
        f.write(content)  # 读取文件

def json_to_instance(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    return instance

def instance_to_json(instance, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        content = json.dumps(instance, ensure_ascii=False, indent=2)
        f.write(content)

def route_save_condition(react_chain, filename):
    substrings = react_chain.split('.')
    all_matched = True

    for substring in substrings:
        grep_command = ['grep', '-qxF', substring, filename]
        result = subprocess.run(grep_command)
        
        if result.returncode != 0:
            all_matched = False
            break
    return all_matched
