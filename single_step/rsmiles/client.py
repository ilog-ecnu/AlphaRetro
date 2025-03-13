# client.py
import requests

def run_client(data):
    url = 'http://localhost:8009'
    response = requests.post(url, json=data)
    
    if response.status_code == 200:  # 200 表示请求成功
        result = response.json()
        print(result['result'])
    else:
        print('Error:', response.status_code)

if __name__ == '__main__':
    input_text = "Cc1nc(-c2ccc(O)c(C#N)c2)sc1C(=O)OCC"
    gpu_id = 3
    beam_size = 10
    data = {
        'input_text':input_text,
        'gpu_id':gpu_id,
        'beam_size': beam_size
    }
    run_client(data)