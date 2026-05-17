from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch import argmax, cat


class RuBertPredictor:
    model_name = "models/best_model"
    num_classes = 3
    id2label = {
    0: "paragraph",
    1: "heading", 
    2: "caption"
    }

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        self.model.eval()
        
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, truncation=False, return_tensors="pt")

        if len(inputs["input_ids"]) > 512:
            for key in inputs:
                inputs[key] = cat([inputs[key][:, :128], inputs[key][:, -386:]], dim=1)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        logits = self.model(input_ids, attention_mask).logits
        result = int(argmax(logits).item())
        return self.id2label[result]
