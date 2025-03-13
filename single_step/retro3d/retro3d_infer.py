import torch
import torch.nn.functional as F
import pickle

from utils import get_saved_info, get_pretrained_model, SequenceGenerator
from utils.smiles_utils import smi_tokenizer
from utils.smiles_graph import SmilesGraph
from utils.smiles_threed import SmilesThreeD


class DataWrap:
    def __init__(self, vocab_path='single_step/retro3d/ckpt/vocab.pk'):
        self.vocab_i2t, self.src_t2i = self.read_vocab(vocab_path)

    def read_vocab(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            self.src_i2t, self.tgt_i2t = pickle.load(f)
        self.src_t2i = {self.src_i2t[i]: i for i in range(len(self.src_i2t))}
        self.tgt_t2i = {self.tgt_i2t[i]: i for i in range(len(self.tgt_i2t))}
        return self.src_i2t, self.src_t2i

    def reconstruct_smi(self, indexs):
        illgel_words = ['<pad>', '<sos>', '<eos>', '<UNK>'] + ['<RX_{}>'.format(i) for i in range(1, 11)]
        illgel_index = [self.src_t2i[word] for word in illgel_words]
        return [self.vocab_i2t[i] for i in indexs if i not in illgel_index]

    def process_data(self, smi, react_type):

        mol_graph = SmilesGraph(smi)
        mol_threed = SmilesThreeD(smi)
        if mol_threed.atoms_coord is None:
            raise KeyError(f"[RDKit] Generate Conformer Error: {smi}")

        bond_matrix = mol_graph.adjacency_matrix_attr
        dist_matrix = mol_threed.dist_matrix

        atoms_coord = mol_threed.atoms_coord
        atoms_index = mol_threed.atoms_index
        atoms_token = [self.src_t2i.get(at, self.src_t2i['<unk>']) for at in mol_threed.atoms_token]
        
        # prepare input
        smi_seq = []
        smi_list = [react_type] + smi_tokenizer(smi)
        smi_list = [self.src_t2i.get(st, self.src_t2i['<unk>']) for st in smi_list]
        smi_seq.append(smi_list)
        smi_tensor = torch.as_tensor(smi_seq, dtype=torch.long)

        bond_matrix = torch.from_numpy(bond_matrix)
        bond_matrix = F.pad(bond_matrix, pad=(0, 0, 1, 0, 1, 0), mode='constant', value=0)
        bond_matrix = bond_matrix.unsqueeze(0).repeat(1, 1, 1, 1)

        dist_matrix = torch.from_numpy(dist_matrix)
        dist_matrix = F.pad(dist_matrix, pad=(1, 0, 1, 0), mode='constant', value=0)
        dist_matrix = dist_matrix.unsqueeze(0).repeat(1, 1, 1)

        batch_index = torch.arange(1).repeat_interleave(len(atoms_index))
        atoms_index = torch.as_tensor(atoms_index).to(torch.long).repeat(1)
        atoms_token = torch.as_tensor(atoms_token).to(torch.long).repeat(1)
        atoms_coord = torch.as_tensor(atoms_coord).to(torch.float32).repeat(1, 1)
        
        return smi_tensor, bond_matrix, dist_matrix, (atoms_coord, atoms_token, atoms_index, batch_index)


def Retro3d_Inference(tgt_smi, react_type, beam_size, vocab_path, pretrained_path):

    # Load pretrained wieght and config
    data_wrap = DataWrap(vocab_path=vocab_path)
    config, model_dict = get_saved_info(pretrained_path)
    model = get_pretrained_model(config, model_dict, data_wrap)
    reaction_predictor = SequenceGenerator(config, model, data=data_wrap, beam_size=beam_size)
    reaction_predictor = reaction_predictor.eval().cuda()

    smi_tensor, bond_matrix, dist_matrix, atoms_attr = data_wrap.process_data(tgt_smi, react_type)
    atoms_coord, atoms_token, atoms_index, batch_index = atoms_attr
    
    smi_tensor, bond_matrix, dist_matrix = \
            smi_tensor.cuda(), bond_matrix.cuda(), dist_matrix.cuda()
    atoms_coord, atoms_token, atoms_index, batch_index = \
        atoms_coord.cuda(), atoms_token.cuda(), atoms_index.cuda(), batch_index.cuda()

    hyposis = reaction_predictor.forward((smi_tensor, bond_matrix, dist_matrix, \
                                        atoms_coord, atoms_token, atoms_index, batch_index))

    pred_results = hyposis[0]   #[i][j]['tokens', 'score', 'alignment', 'positional_scores']
    smiles_result = []
    scores_result = []
    for j in range(beam_size):
        pred_tokens = pred_results[j]['tokens'].cpu().numpy().tolist()
        pred_smiles = ''.join(data_wrap.reconstruct_smi(pred_tokens))
        pred_scores = pred_results[j]['score'].cpu().numpy()
        smiles_result.append(pred_smiles.strip())
        scores_result.append(pred_scores.item())
    return smiles_result, scores_result