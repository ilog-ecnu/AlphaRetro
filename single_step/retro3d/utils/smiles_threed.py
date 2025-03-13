from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import re

from .smiles_utils import smi_tokenizer, clear_map_smiles

ATOMTYPES = ['UNK', 'C', 'H', 'O', 'N', 'S', 'Li', 'Mg', 'F', 'K', 'B', 'Cl', \
'I', 'Se', 'Si', 'Sn', 'P', 'Br', 'Zn', 'Cu', 'Pt', 'Fe', 'Pd', 'Pb']
ATOMTOI = {atom: i for i, atom in enumerate(ATOMTYPES)}


class SmilesThreeD:
    def __init__(self, rooted_smi, before=None, existing=None):
        if existing is not None:
            assert len(existing) == 3
            self.atoms_coord, self.atoms_token, self.atoms_index = existing
        else:
            self.atoms_coord = self.get_atoms_coordinate(rooted_smi, before)
            self.atoms_token, self.atoms_index = self.get_atoms_dict(rooted_smi)

        V = len(smi_tokenizer(rooted_smi))
        self.dist_matrix = np.zeros((V, V), dtype=float)
        if self.atoms_coord is not None:
            atoms_coord = np.array(self.atoms_coord)
            for i, index_i in enumerate(self.atoms_index):
                for j, index_j in enumerate(self.atoms_index):
                    self.dist_matrix[index_i, index_j] = \
                        np.linalg.norm(atoms_coord[i]-atoms_coord[j])
        
    def get_atoms_dict(self, smi):
        smi = clear_map_smiles(smi)
        # atoms_token = [ATOMTOI.get(atom.GetSymbol(), 0) for atom in Chem.MolFromSmiles(smi).GetAtoms()]
        # atoms_index = [i for i, token in enumerate(smi_tokenizer(smi)) if any(c.isalpha() for c in token)]
        atoms_index, atoms_token = zip(*[(i, token) for i, token in enumerate(smi_tokenizer(smi)) if any(c.isalpha() for c in token)])
        return atoms_token, atoms_index

    def get_atoms_coordinate(self, rooted_smi, before=None):
        if before is not None:  # re-sort coordinates instead of re-computing
            old_smi, atoms_coord = before
            smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", old_smi)))
            rooted_smi_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", rooted_smi)))
            positions = [atoms_coord[smi_map_numbers.index(i)] for i in rooted_smi_map_numbers]
        else:
            mol = Chem.MolFromSmiles(rooted_smi)
            if mol.GetNumAtoms() < 2:
                return None
            mol = Chem.AddHs(mol)
            ignore_flag1 = 0
            while AllChem.EmbedMolecule(mol, randomSeed=10) == -1:
                ignore_flag1 = ignore_flag1 + 1
                if ignore_flag1 >= 10:
                    return None
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            positions = mol.GetConformer().GetPositions().tolist()
        return positions
