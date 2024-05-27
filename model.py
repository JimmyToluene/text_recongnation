import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd
import nltk
from pypinyin import lazy_pinyin


nltk.download('punkt')

# 示例数据准备
data = {
    'question': [
        "计算NUM3列表的A列总和",
        "计算sheet2下一个表的a列总和",
        "求number2的平均值",
        "计算sheet1后面的表的B列"
    ],
    'sheet': [
        "number3",
        "sheet3",
        "number2",
        "sheet2"
    ]
}

df = pd.DataFrame(data)

# 构造二分类数据集
sheets = df['sheet'].unique()
data = []
for _, row in df.iterrows():
    question = row['question']
    correct_sheet = row['sheet']
    for sheet in sheets:
        label = 1 if sheet == correct_sheet else 0
        data.append({'question': question, 'sheet': sheet, 'label': label})

data_df = pd.DataFrame(data)
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})


def preprocess_text(text):
    # 标准化输入格式：处理大小写，去除无关字符
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # 处理缩写、同音词、形近词等（此处示例简单替换，可以根据具体需求扩展）
    replacements = {
        'num': 'number',
        'sheet': 'sheet'
    }

    words = text.split()
    text = ' '.join([replacements.get(word, word) for word in words])

    return text


def preprocess_function(examples):
    questions, sheets = [], []
    for question, sheet in zip(examples['question'], examples['sheet']):
        preprocessed_question = preprocess_text(question)
        preprocessed_sheet = preprocess_text(sheet)
        questions.append(preprocessed_question)
        sheets.append(preprocessed_sheet)

    return tokenizer(questions, sheets, padding='max_length', truncation=True)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_dataset = dataset.map(preprocess_function, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
)

trainer.train()


def predict(question, sheets):
    preprocessed_question = preprocess_text(question)
    best_match = None
    best_score = -float('inf')

    for sheet in sheets:
        preprocessed_sheet = preprocess_text(sheet)
        inputs = tokenizer(preprocessed_question, preprocessed_sheet, return_tensors='pt', padding=True,
                           truncation=True)
        outputs = model(**inputs)
        score = outputs.logits[0][1].item()  # 得分越高，匹配可能性越大

        if score > best_score:
            best_score = score
            best_match = sheet

    return best_match


# 示例预测
sheets = df['sheet'].unique()
question = "计算NUM3列表的A列总和"
print(predict(question, sheets))

