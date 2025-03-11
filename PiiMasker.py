import os
import time
import torch
import tkinter as tk
from tkinter import Menu, filedialog, messagebox, scrolledtext
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# Подключаем обученную модель
class BERT_CRF(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import BertModel
        from torchcrf import CRF

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))

        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask)
            return loss.mean()
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

# Загрузка модели
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("bert_crf_model.pth", map_location=DEVICE)
tag2id = checkpoint['tag2id']
id2tag = checkpoint['id2tag']
model = BERT_CRF("bert-base-multilingual-cased", num_labels=len(tag2id)).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Функция предсказания PII в тексте
def mask_pii(text, mask_mode="all"):  # mask_mode: "all" -> ***, "partial" -> A***
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    
    masked_text = []
    pii_count = 0

    for token, label_id in zip(tokens, predictions):
        label = id2tag[label_id[0]]
        
        if label.startswith("B-") or label.startswith("I-"):
            pii_count += 1
            if mask_mode == "all":
                masked_text.append("*" * len(token))
            else:
                masked_text.append(token[0] + "*" * (len(token) - 1))
        else:
            masked_text.append(token)

    return tokenizer.convert_tokens_to_string(masked_text), pii_count

# GUI с использованием Tkinter
class PiiMaskerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PII Маскер")
        self.root.geometry("500x400")

        # Кнопка выбора файлов
        self.select_button = tk.Button(root, text="Выбрать файлы", command=self.select_files)
        self.select_button.pack(pady=5)

        # Выбор режима маскирования
        self.mask_mode_label = tk.Label(root, text="Режим маскирования:")
        self.mask_mode_label.pack()
        self.mask_mode_var = tk.StringVar(value="all")
        self.mask_mode_dropdown = tk.OptionMenu(root, self.mask_mode_var, "all", "partial")
        self.mask_mode_dropdown.pack()

        # Флажок: перезаписывать или нет
        self.overwrite_var = tk.BooleanVar()
        self.overwrite_checkbox = tk.Checkbutton(root, text="Перезаписывать файлы", variable=self.overwrite_var)
        self.overwrite_checkbox.pack()

        # Кнопка обработки файлов
        self.process_button = tk.Button(root, text="Обработать файлы", command=self.process_files)
        self.process_button.pack(pady=5)

        # Поле для логов
        self.log_output = scrolledtext.ScrolledText(root, width=60, height=10, wrap=tk.WORD)
        self.log_output.pack(pady=5)
        self.log_output.bind("<Button-3>", self.show_context_menu)

        self.context_menu = Menu(root, tearoff=0)
        self.context_menu.add_command(label="Копировать", command=self.copy_text)


        self.selected_files = []

    def copy_text(self):
        try:
            selected_text = self.log_output.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.root.update()
        except tk.TclError:
            pass  # Если ничего не выделено, ничего не делать

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def select_files(self):
        files = filedialog.askopenfilenames(title="Выберите текстовые файлы", filetypes=[("Text Files", "*.txt")])
        if files:
            self.selected_files = list(files)
            self.log_output.insert(tk.END, f"Выбраны файлы: {', '.join(os.path.basename(f) for f in files)}\n")

    def process_files(self):
        if not self.selected_files:
            messagebox.showwarning("Ошибка", "⚠️ Сначала выберите файлы!")
            return
        
        mask_mode = self.mask_mode_var.get()
        overwrite = self.overwrite_var.get()

        for file_path in self.selected_files:
            start_time = time.time()
            with open(file_path, "r", encoding="utf-8") as infile:
                text = infile.read()
            
            masked_text, pii_count = mask_pii(text, mask_mode)
            
            word_count = len(text.split())
            save_path = file_path if overwrite else file_path.replace(".txt", "_masked.txt")
            
            with open(save_path, "w", encoding="utf-8") as outfile:
                outfile.write(masked_text)
            
            elapsed_time = round(time.time() - start_time, 2)

            log_message = (f"✅ {os.path.basename(file_path)} обработан\n"
                           f"📄 Сохранен в: {save_path}\n"
                           f"📊 Всего слов: {word_count}, Замаскировано: {pii_count}\n"
                           f"⏱ Время обработки: {elapsed_time} сек\n\n")
            self.log_output.insert(tk.END, log_message)
            self.log_output.yview(tk.END)  # Автоскролл вниз

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = PiiMaskerGUI(root)
    root.mainloop()
