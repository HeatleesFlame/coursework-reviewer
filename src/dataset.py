import torch
from torch.utils.data import Dataset

class DocumentDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if item["is_negative"] == True:
            item["label_id"] = 0
        text = item["text"]
        inputs = self.tokenizer(text, truncation=False)

        if len(inputs["input_ids"]) > 512:
            for key in inputs:
                inputs[key] = inputs[key][:128] + inputs[key][-386:]

        inputs["label"] = item["label_id"]      
        for key in inputs:
            inputs[key] = torch.tensor(inputs[key])
        
        return inputs