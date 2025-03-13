import subprocess
import tempfile
import os
import re
from rdkit import Chem

def cano(smiles):
    """标准化判定"""
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi

def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def translate_with_opennmt(input_smi, model_path, beam_size):
    try:
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_input:
            temp_input.write(smi_tokenizer(input_smi))
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(delete=False, mode='r', encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name

        command = [
            "onmt_translate",
            "-model", model_path,
            "-src", temp_input_path,
            "-output", temp_output_path,
            "-gpu", str(0),
            "-beam_size", str(beam_size),
            "-n_best", str(beam_size),
            "-batch_size", "8192",
            "-batch_type", "tokens",
            "-max_length", "500",
            "-seed", "0"
        ]

        with open(os.devnull, "w") as devnull:
            subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

        with open(temp_output_path, "r", encoding="utf-8") as output_file:
            results = output_file.readlines()

        return [line.strip().replace(' ', '') for line in results]

    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def main(prod_smi, model_path="single_step/rsmiles/model/USPTO_full_PtoR.pt", beam_size=10):
    results = translate_with_opennmt(cano(prod_smi), model_path, beam_size)
    return results


if __name__ == "__main__":
    prod_smi = "Cc1nc(-c2ccc(O)c(C#N)c2)sc1C(=O)OCC"
    results = main(prod_smi, beam_size=10)
    for i, result in enumerate(results):
        print(f"Top {i + 1}: {result}")
