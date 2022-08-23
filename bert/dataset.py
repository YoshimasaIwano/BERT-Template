from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, training):
        super(BertDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        txt = self.data["description"].iloc[index] 
        
        inputs = self.tokenizer.encode_plus(
            txt, 
            None,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        if self.training:
            target = torch.tensor(self.data["jobflag"].iloc[index])
            target = F.one_hot(target, num_classes=4).float()
            return ids, mask, token_type_ids, target
        else:
            return ids, mask, token_type_ids