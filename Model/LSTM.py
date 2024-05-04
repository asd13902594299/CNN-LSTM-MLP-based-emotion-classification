import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout):
        
        super().__init__()
                
        self.rnn = nn.LSTM(embedding_dim,  # input_size
                           hidden_dim,  # output_size
                           num_layers=n_layers,  # 层数
                           bidirectional=bidirectional, # 是否双向
                           dropout=dropout) # 随机去除神经元
        
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.tensor, text_len: torch.tensor):
        """
        # text.shape = [batch_size, height, width]
        # text_len.shape = [batch_size]
        text: tensor, text_len: tensor (input)


        # Pack the sentence for better performance and memory usage
        # packed_text.data.shape = [sum(text_len), width]
        packed_text = nn.utils.rnn.pack_padded_sequence(input=text, lengths=text_len.cpu(), batch_first=True, enforce_sorted=False)


        # Equivalent to two LSTM layers
        # packed_output.shape = [sum(text_len), hid_dim * num_directions]
        # hidden.shape = [num_layers * num_directions, batch_size, hid_dim]
        # cell.shape = [num_layers * num_directions, batch_size, hid_dim]
        packed_output, (hidden, cell) = self.rnn(packed_text)


        # Only the last forward and backward hidden layers are needed
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))


        # fc(hidden) = [1]
        return self.fc(hidden) (output)
        """
        
        
        # text.shape = [batch_size, height, width]
        # text_len.shape = [batch_size]
        
        packed_text = nn.utils.rnn.pack_padded_sequence(input=text, lengths=text_len.cpu(), batch_first=True, enforce_sorted=False)
        
        # packed_text.data.shape = [sum(text_len), width]
        
        packed_output, (hidden, cell) = self.rnn(packed_text)
        
        # packed_output.shape = [sum(text_len), hid_dim * num_directions]
        # hidden.shape = [num_layers * num_directions, batch_size, hid_dim]
        # cell.shape = [num_layers * num_directions, batch_size, hid_dim]
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.fc(hidden)

