import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torchcrf import CRF
from seqeval.metrics import classification_report
from conllu_import import read_nerus_conllu_limited

####################################
# ПРЕДОБРАБОТКА ДАННЫХ: Токенизация и сохранение меток
####################################
def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Токенизирует предложение с помощью BertTokenizer и корректно сохраняет метки.
    При токенизации сложных слов первый токен получает префикс B-, остальные – I-.
    """
    tokenized_sentence = []
    new_labels = []

    for word, label in zip(sentence, text_labels):
        # Убираем возможные префиксы из исходной метки
        #label = label.replace("Tag=", "").replace("g=", "")
        tokenized_word = tokenizer.tokenize(word)
        if not tokenized_word:
            continue  # пропускаем пустые токены
        tokenized_sentence.extend(tokenized_word)
        main_label = label if label == "O" else "B-" + label.split("-")[-1]
        new_labels.append(main_label)
        new_labels.extend(["I-" + label.split("-")[-1] if label != "O" else "O"] * (len(tokenized_word) - 1))
    return tokenized_sentence, new_labels

def encode_tags(label_seqs, tag2id):
    """
    Преобразует последовательности меток в последовательности индексов.
    """
    return [[tag2id[tag] for tag in seq] for seq in label_seqs]

####################################
# ПОДГОТОВКА ДАННЫХ: Разделение, токенизация и формирование DataLoader
####################################
def prepare_data(nerus_path, tokenizer, max_samples=10000, max_len=32):
    # Читаем данные
    data = read_nerus_conllu_limited(nerus_path, max_samples=max_samples)
    #data = list(zip(sentences, labels))
    random.shuffle(data)
    
    # Разделяем данные: 80% для обучения и 20% для теста
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_sentences, train_labels = zip(*train_data)
    test_sentences, test_labels = zip(*test_data)
    
    # Токенизируем данные с сохранением меток
    train_tokens, train_labels_tok = zip(*[
        tokenize_and_preserve_labels(sent, lbl, tokenizer)
        for sent, lbl in zip(train_sentences, train_labels)
    ])
    test_tokens, test_labels_tok = zip(*[
        tokenize_and_preserve_labels(sent, lbl, tokenizer)
        for sent, lbl in zip(test_sentences, test_labels)
    ])
    
    # Создаём словарь меток из тренировочной выборки
    unique_tags = set(tag for seq in train_labels_tok for tag in seq)
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    
    # Кодируем метки
    train_tag_ids = encode_tags(train_labels_tok, tag2id)
    test_tag_ids = encode_tags(test_labels_tok, tag2id)
    
    # Преобразуем список меток в тензоры и выполняем паддинг до max_len
    train_tag_tensors = [torch.tensor(seq, dtype=torch.long) for seq in train_tag_ids]
    test_tag_tensors = [torch.tensor(seq, dtype=torch.long) for seq in test_tag_ids]
    train_tag_padded = pad_sequence(train_tag_tensors, batch_first=True, padding_value=tag2id["O"])[:, :max_len]
    test_tag_padded = pad_sequence(test_tag_tensors, batch_first=True, padding_value=tag2id["O"])[:, :max_len]
    
    # Для входных данных используем готовый токенизатор BERT с паддингом и усечением
    train_encoded = tokenizer(
        [" ".join(sent) for sent in train_sentences],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    test_encoded = tokenizer(
        [" ".join(sent) for sent in test_sentences],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # Собираем датасеты
    train_dataset = TensorDataset(train_encoded["input_ids"], train_encoded["attention_mask"], train_tag_padded)
    test_dataset = TensorDataset(test_encoded["input_ids"], test_encoded["attention_mask"], test_tag_padded)
    
    return train_dataset, test_dataset, tag2id, id2tag

####################################
# МОДЕЛЬ BERT+CRF
####################################
class BERT_CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BERT_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        
        if labels is not None:
            # Маска для CRF (только активные токены)
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask)
            return loss.mean()  # Усредняем значение loss
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

####################################
# ОБУЧЕНИЕ МОДЕЛИ
####################################
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

####################################
# ОЦЕНКА МОДЕЛИ
####################################
def evaluate_model(model, test_loader, device, id2tag):
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            predictions = model(input_ids, attention_mask)
            
            # Выравнивание предсказаний и истинных меток (игнорируем паддинг)
            for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
                true_labels = []
                pred_labels = []
                for p, l, m in zip(pred_seq, label_seq, mask_seq):
                    if m.item() == 1:
                        true_labels.append(id2tag[l.item()])
                        pred_labels.append(id2tag[p])
                all_true.append(true_labels)
                all_preds.append(pred_labels)
                
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds))