# NLP模型探索
# 根据官网指引，从0开始搭建自然语言处理模型
# 1. 从零开始的自然语言处理：使用字符级的RNN模型分类名称
# 2. 从零开始的自然语言处理：使用字符级 RNN 生成姓名
# 3. 从零开始的自然语言处理：利用序列到序列网络和注意力机制进行翻译
# 然而由于torchtext已经停止更新，要使用nlp模型，可以使用第三方库，如Hugging Face的transformers库，这些库提供了大量预训练模型，包括但不限于BERT、GPT、RoBERTa等。

# BERT（Bidirectional Encoder Representations from Transformers），BERT模型架构是的NLP中各类任务也可以使用预训练+微调范式（类似CV领域）
# 1 . 先进行预训练得到基础模型
# 2 . 再在特定任务上对基础模型进行微调

# 如：使用google的bert结合哈工大预训练模型进行中文/英文文本二分类，基于pytorch和transformer
# from transformers import BertTokenizer
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# # Load the BERT tokenizer.
# print('Loading BERT tokenizer...')
# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
# 这里导入的模型是hfl/chinese-bert-wwm-ext
# 参考：https://blog.csdn.net/Jerryzhangjy/article/details/110209984

# 代码演示：基于中文文本预训练的BERT基础模型训练文本分类模型
# 参考： https://blog.csdn.net/WSSsharon/article/details/146497452
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
 
# 设置模型路径
MODEL_PATH = './bert-base-chinese'  # 本地模型路径
 
# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
 
# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
 
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
 
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
 
# BERT分类器模型
class BertClassifier(nn.Module):
    def __init__(self, model_path, n_classes=2):
        # super(BertClassifier, self).__init__() python2写法
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
 
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
 
def train_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
 
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
        # 验证
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch + 1}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # 只保存模型权重
            torch.save(model.state_dict(), 'best_model.pth')
 
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions)
    recall = recall_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)
    
    print(f'\n评估指标:')
    print(f'准确率 (Accuracy): {accuracy:.4f}')
    print(f'精确率 (Precision): {precision:.4f}')
    print(f'召回率 (Recall): {recall:.4f}')
    print(f'F1分数 (F1-Score): {f1:.4f}')
    
    return accuracy
 
def main():
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：未找到本地模型文件夹 '{MODEL_PATH}'")
        print("请先下载模型到本地，或修改 MODEL_PATH 为正确的模型路径")
        return
 
    # 设置设备和随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    set_seed(42)
    
    # 加载数据
    df = pd.read_excel('data4.xlsx')  # 确保Excel文件包含'text'和'label'列
    texts = df['text'].values
    labels = df['label'].values
    
    # 划分训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # 初始化tokenizer和数据集
    print("加载本地tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 初始化模型
    print("加载本地预训练模型...")
    model = BertClassifier(MODEL_PATH)
    model.to(device)
    
    # 训练模型
    print("开始训练...")
    train_model(model, train_loader, test_loader, device)
    
    # 加载最佳模型并评估
    print("\n加载最佳模型并进行最终评估...")
    # 使用weights_only=True安全加载模型权重
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    evaluate_model(model, test_loader, device)
 
if __name__ == '__main__':
    main() 