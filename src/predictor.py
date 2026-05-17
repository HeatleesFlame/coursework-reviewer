import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RuBertPredictor:
    model_name = "models/best_model"
    num_classes = 3
    id2label = {
        0: "paragraph",
        1: "heading",
        2: "caption",
    }

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
        )
        self.model.eval()

    def _truncate(self, encoding: dict) -> dict:
        """Keep first 128 + last 384 tokens when sequence exceeds 512."""
        if len(encoding["input_ids"]) > 512:
            return {key: encoding[key][:128] + encoding[key][-384:] for key in encoding}
        return encoding

    def predict(self, text: str) -> str:
        enc = self._truncate(self.tokenizer(text, truncation=False))
        input_ids = torch.tensor([enc["input_ids"]])
        attention_mask = torch.tensor([enc["attention_mask"]])
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask).logits
        return self.id2label[int(torch.argmax(logits).item())]

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[str]:
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [self._truncate(self.tokenizer(t, truncation=False)) for t in texts[i:i + batch_size]]

                max_len = max(len(e["input_ids"]) for e in batch)
                pad_id = self.tokenizer.pad_token_id

                input_ids = torch.tensor([
                    e["input_ids"] + [pad_id] * (max_len - len(e["input_ids"])) for e in batch
                ])
                attention_mask = torch.tensor([
                    e["attention_mask"] + [0] * (max_len - len(e["attention_mask"])) for e in batch
                ])

                logits = self.model(input_ids, attention_mask).logits
                results.extend(self.id2label[p] for p in torch.argmax(logits, dim=1).tolist())

        return results
