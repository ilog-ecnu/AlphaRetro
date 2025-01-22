# client.py
import requests

def run_client(data):
    url = 'http://localhost:8010'
    response = requests.post(url, json=data)
    
    if response.status_code == 200:  # 200 表示请求成功
        result = response.json()
        # print(result)
        print(result['result'])
        print(result['score'])
    else:
        print('Error:', response.status_code)
    return result

if __name__ == '__main__':
    input_text = "<RX_3>CC(C)(C)OC(=O)NCc1cc2ccccc2nc1C#C[Si](C)(C)C"
    gpu_id = 3
    beam_size = 10
    data = {
        'input_text':input_text,
        'gpu_id':gpu_id,
        'beam_size': beam_size
    }
    run_client(data)