import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim

# 示例数据集
questions = [
    "计算NUM3列表的A列总和",
    "计算sheet2下一个表的a列总和",
    "计算nuber3列表的A列总和",
    "计算不存在的表的A列总和"
]
sheet_lists = [
    ["number1", "number2", "number3"],
    ["sheet1", "sheet2", "sheet3"],
    ["number1", "number2", "number3"],
    ["number1", "number2", "number3"]
]
labels = ["number3", "sheet3", "number3", "err_mismatch"]

# 加载预训练的BERT tokenizer 和 model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')


# 编码函数
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


# 编码数据
encoded_questions = [encode_text(q) for q in questions]
encoded_sheets = [encode_text(' '.join(sheets)) for sheets in sheet_lists]
encoded_data = [torch.cat((q, s), dim=1) for q, s in zip(encoded_questions, encoded_sheets)]
encoded_labels = [sheet_lists[i].index(labels[i]) if labels[i] in sheet_lists[i] else len(sheet_lists[i]) for i in
                  range(len(labels))]

# 转换为Tensor
data_tensor = torch.cat(encoded_data)
label_tensor = torch.tensor(encoded_labels)


# 设计一个简单的神经网络作为分类器
class MatchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MatchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_dim = 768 * 2  # BERT base model output dimension for question + sheet list
hidden_dim = 512
output_dim = max([len(sheets) for sheets in sheet_lists]) + 1  # max sheet list length + 1 for err_mismatch

model = MatchModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(data_tensor)
    loss = criterion(outputs, label_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 使用训练好的模型进行预测
def predict(question, sheet_list):
    model.eval()
    encoded_question = encode_text(question)
    encoded_sheet_list = encode_text(' '.join(sheet_list))
    encoded_input = torch.cat((encoded_question, encoded_sheet_list), dim=1)
    output = model(encoded_input)
    _, predicted = torch.max(output.data, 1)
    if predicted.item() < len(sheet_list):
        return sheet_list[predicted.item()]
    else:
        return "err_mismatch"


# 示例预测
question = "计算nuber3列表的A列总和"
sheet_list = ["number1", "number2", "number3"]
print(predict(question, sheet_list))  # 应输出：number3
