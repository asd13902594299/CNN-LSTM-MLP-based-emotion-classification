import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, sentence_len, embedding_dim, truncate_len, hidden_dim, output_dim, 
                 dropout):
        
        super().__init__()
        
        self.truncate_len = truncate_len
        
        self.fc_in = nn.Linear(truncate_len*embedding_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.tensor, text_len):
        """
        # text.shape = [batch_size, height, width]
        text: tensor (input)

        # text.shape = [batch_size, truncate_len, width]
        text = text[:, :self.truncate_len, :]
        

        # text.shape = [batch_size, truncate_len*width]
        text = text.view(text.shape[0], -1)


        # fc_in.shape = [batch_size, hidden_dim]
        fc_in = F.relu(self.fc_in(text))
        
        
        # fc_h1.shape = [batch_size, hidden_dim]
        fc_h1 = F.relu(self.fc_hidden(fc_in))
        
        
        # fc_h2.shape = [batch_size, hidden_dim]
        fc_h2 = self.dropout(F.relu(self.fc_hidden(fc_h1)))

        
        # fc_out.shape = [batch_size, output_dim]
        fc_out = self.fc_out(fc_h2)
        
        
        return fc_out
        """
        
        # text.shape = [batch_size, height, width]
        
        text = text[:, :self.truncate_len, :]
        
        # text.shape = [batch_size, truncate_len, width]
        
        text = text.view(text.shape[0], -1)

        # text.shape = [batch_size, truncate_len*width]
        
        text = F.relu(self.fc_in(text))
        
        # fc_in.shape = [batch_size, hidden_dim]
        
        text = F.relu(self.fc_hidden(text))
        
        # fc_h1.shape = [batch_size, hidden_dim]
        
        text = self.dropout(F.relu(self.fc_hidden(text)))
        
        # fc_h2.shape = [batch_size, hidden_dim]
        
        text = self.fc_out(text)
        
        # fc_out.shape = [batch_size, output_dim]
        
        return text

