import torch.nn as nn
from transformers import BertModel

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.fc = nn.Linear(768, 768)
        self.final_layer = nn.Linear(768, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, ids, mask, token_type_ids):
        _, out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
#         out = self.fc(out_bert)
#         out = self.final_layer(out+out_bert) # skipping connection
        out = self.final_layer(out)
        out = self.softmax(out)
        
        return out