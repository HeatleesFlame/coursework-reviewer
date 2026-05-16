from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from dataset import DocumentDataset


MODEL_NAME = "cointegrated/rubert-tiny2"
DATA_DIR = r"C:\Users\User\Documents\VScode projects\coursework-reviewer\data\processed"
NUM_CLASSES = 3


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer, max_length=510)

    train_dataset = DocumentDataset(load_dataset("json", data_dir=DATA_DIR, split="train"), tokenizer)
    val_dataset = DocumentDataset(load_dataset("json", data_dir=DATA_DIR, split="validation"), tokenizer)
    test_dataset = DocumentDataset(load_dataset("json", data_dir=DATA_DIR, split="test"), tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    train_labels = [i["label"].item() for i in train_dataset]
    classes = np.unique(train_labels)
    class_weights = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    ))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.float().to(device))

    epochs = 3
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.1 * total_steps)
    best_f1 = 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1} ---")

        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in progress_bar:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

            loss = loss_fn(outputs.logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)

                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(b_labels.detach().cpu().numpy())

        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            model.save_pretrained("models/best_model")
            tokenizer.save_pretrained("models/best_model")
            print(f"Модель сохранена с F1: {best_f1:.4f}")
        print(f"Epoch Validation F1 Macro: {val_f1_macro:.4f}")
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        print(f"F1 по классам: paragraph={f1_per_class[0]:.4f}, heading={f1_per_class[1]:.4f}, caption={f1_per_class[2]:.4f}")
