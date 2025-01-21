import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer
from torch import nn, optim
from transformers import DistilBertModel


class SmilesClassifier(nn.Module):
    def __init__(self, num_labels=10, multi_label=True):
        super(SmilesClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.multi_label = multi_label

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(outputs[0][:,0])
        return self.output(output)

    def predict(self, input_ids, attention_mask):
        logits = self(input_ids, attention_mask)
        if self.multi_label:
            # 多标签：使用sigmoid激活
            probs = torch.sigmoid(logits)
            return (probs > 0.5).int()
        else:
            # 单标签：使用softmax激活
            probs = torch.softmax(logits, dim=1)
            return torch.argmax(probs, dim=1)


class SmilesDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256, multi_label=True):
        self.tokenizer = tokenizer
        self.smiles = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                smile, labels = line.strip().split('>>')
                if multi_label:
                    label_list = list(map(int, labels.split(',')))
                    label_vector = [1 if i in label_list else 0 for i in range(1, 11)]
                    self.labels.append(label_vector)
                else:
                    self.labels.append(int(labels) - 1)  # 为单标签任务减1以适配从0开始的类别索引
                self.smiles.append(smile)
        self.max_len = max_len
        self.multi_label = multi_label

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        labels = self.labels[idx]
        encoded = self.tokenizer.encode_plus(smile, add_special_tokens=True, max_length=self.max_len, 
                                             return_token_type_ids=False, padding='max_length', 
                                             truncation=True, return_attention_mask=True, return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float if self.multi_label else torch.long)
        }

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)

def validate_epoch(model, valid_loader, criterion, device):
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_valid_loss += loss.item()
    return total_valid_loss / len(valid_loader)

def save_model(model, epoch, path='single_step/rxnfp'):
    torch.save(model.state_dict(), f"{path}_{epoch}.pth")

if __name__ == '__main__':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = SmilesClassifier(num_labels=10, multi_label=False)  # 根据需要更改multi_label
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss() if model.multi_label else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # 加载数据并分割成训练集和验证集
    full_dataset = SmilesDataset('data/rxnfp/rxnfp_example.txt', tokenizer, multi_label=model.multi_label)
    train_size = int(0.995 * len(full_dataset))  # 留下大约 1/200 的数据作为验证集
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    num_epochs = 50  # 总的训练轮数
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate_epoch(model, valid_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")
        
        if (epoch + 1) % 10 == 0:
            save_model(model, epoch + 1)
            print(f"Saved model at epoch {epoch + 1}")
