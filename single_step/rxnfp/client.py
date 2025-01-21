# client.py
import requests

def client_rx(data):
    url = 'http://localhost:8100'
    response = requests.post(url, json=data)
    
    if response.status_code == 200:  # 200 表示请求成功
        result = response.json()
    else:
        print('Error:', response.status_code)
    return str(result['result'])

if __name__ == '__main__':
    input_text = "O=C(OCc1ccccc1)[C@H](O)[C@@H](O)c1ccc(OCc2ccccc2)c(OCc2ccccc2)c1"

    data = {
        'input_text':input_text,
    }
    output = client_rx(data)
    print(output)