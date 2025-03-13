# start_server.py
from flask import Flask, request, jsonify
from translate import main
app = Flask(__name__)

@app.route('/', methods=['POST'])
def infer():
    data = request.json
    input_text = data.get('input_text')
    gpu_id = data.get('gpu_id', 1)  # 允许通过请求指定使用哪个 GPU
    beam_size = data.get('beam_size', 10)

    result = main(input_text, beam_size=beam_size)

    return jsonify({'result': result})
	
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009)