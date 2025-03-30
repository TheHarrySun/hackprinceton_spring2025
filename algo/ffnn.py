import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim = 128, combined_hidden = 64, num_classes = 1):
        super(FFNN, self).__init__()
        
        self.drug1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.drug2_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim, combined_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_hidden, num_classes)
        )
        
    def forward(self, drug1_feat, drug2_feat):
        drug1_out = self.drug1_net(drug1_feat)
        drug2_out = self.drug2_net(drug2_feat)
        
        combined_feat = torch.cat((drug1_out, drug2_out), dim=1)
        out = self.combined_net(combined_feat)
        
        return out
