from retro3d_infer import Retro3d_Inference
import re
from rdkit import Chem

def cano(smiles):
    """标准化判定"""
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi

def run_inference(smiles, beam_size, vocab_path, pretrained_path):
    pattern = r"(<RX_\d+>)(.*)"
    match = re.match(pattern, smiles)
    if match:
        react_type = match.group(1)
        smiles = match.group(2)
    smiles_result, scores_result = Retro3d_Inference(cano(smiles), react_type, beam_size, vocab_path, pretrained_path)
    return smiles_result, scores_result

if __name__ == "__main__":
    smiles = '<RX_6>CCC1c2cc3n(c(=O)c2COC1O)Cc1cc2ccccc2nc1-3'
    beam_size = 10
    vocab_path='single_step/retro3d/ckpt/vocab.pk'
    pretrained_path='single_step/retro3d/ckpt/model_camptothecin.chkpt'
    smiles_result, scores_result = run_inference(smiles, beam_size, vocab_path, pretrained_path)
    print(smiles_result)
    print(scores_result)
