from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
    def forward(self, input_text):
        return self.bert(input_text)['pooler_output']

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(768, 4096)
        self.fc2 = nn.Linear(4096, 64*64*3)
        
    def forward(self, text_representation):
        x = torch.relu(self.fc1(text_representation))
        return torch.sigmoid(self.fc2(x)).view(-1, 3, 64, 64)
