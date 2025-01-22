from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from flask import Flask, request, jsonify
from single_step.retro3d.run_inference import run_inference
import re
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


app = Flask(__name__)

def cano(smiles):
    """标准化判定"""
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi

# Set parameters
vocab_path = 'single_step/retro3d/ckpt/vocab.pk'
pretrained_path = 'single_step/retro3d/ckpt/model_camptothecin.chkpt'
# beam_size = 10

@app.route('/', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text')
    gpu_id = data.get('gpu_id', 1)  # 允许通过请求指定使用哪个 GPU
    beam_size = data.get('beam_size', 10)

    smiles_result, scores_result = run_inference(input_text, beam_size, vocab_path, pretrained_path)
    output = []
    for line in smiles_result:
        cano_line = cano(line)
        output.append(cano_line)
    return jsonify({'result': output, 'score': scores_result})
    # print(output)
    # print(scores_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010, threaded=True)