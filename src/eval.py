from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from sklearn.metrics import f1_score, confusion_matrix

from dataset import DocumentDataset


MODEL_NAME = "models/best_model"
DATA_DIR = r"C:\Users\User\Documents\VScode projects\coursework-reviewer\data\processed"
NUM_CLASSES = 3


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer, max_length=510)

    test_dataset = DocumentDataset(load_dataset("json", data_dir=DATA_DIR, split="test").filter(lambda x: not x["is_negative"]), tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=data_collator)

    train_labels = [i["label"].item() for i in test_dataset]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(b_labels.detach().cpu().numpy())

    val_f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch Validation F1 Macro: {val_f1_macro:.4f}")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    print(f"F1 по классам: paragraph={f1_per_class[0]:.4f}, heading={f1_per_class[1]:.4f}, caption={f1_per_class[2]:.4f}")
    print(confusion_matrix(all_labels, all_preds))
