import torch
from transformers import DistilBertTokenizer
from single_step.rxnfp.train import SmilesClassifier
        
def predict_smiles(smiles, model, tokenizer):
    encoded = tokenizer.encode_plus(smiles, add_special_tokens=True, max_length=256, 
                                    return_token_type_ids=False, padding='max_length', 
                                    truncation=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    with torch.no_grad():
        predictions = model.predict(input_ids, attention_mask)
        if model.multi_label:
            labels_indices = [i + 1 for i in range(predictions.shape[1]) if predictions[0, i].item() == 1]
        else:
            labels_indices = predictions.item() + 1  # 加1因为标签从1开始编号

    return labels_indices

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = SmilesClassifier(num_labels=10, multi_label=False)  # 单标签模式，更改为True为多标签
model.load_state_dict(torch.load('single_step/rxnfp/model.pth', map_location=torch.device('cpu')))
model.eval()

# 输入SMILES表达式
smiles_input = "O=Cc1ccc(OCc2ccccc2)c(OCc2ccccc2)c1"

# 获取预测结果
predicted_labels_indices = predict_smiles(smiles_input, model, tokenizer)
print("Predicted Labels:", predicted_labels_indices)