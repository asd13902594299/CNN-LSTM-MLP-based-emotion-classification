import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
                
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec.vectors), freeze=False)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text: torch.tensor, text_len):
        """
        # text.shape = [batch_size, height, width]
        text: tensor (input)


        # text.shape = [batch_size, channels, height, width]
        text = text.unsqueeze(1)


        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        conved_0 = self.dropout(F.relu(self.convs[0](text))).squeeze(3)
        conved_1 = self.dropout(F.relu(self.convs[1](text))).squeeze(3)
        conved_2 = self.dropout(F.relu(self.convs[2](text))).squeeze(3)
        conved_3 = self.dropout(F.relu(self.convs[3](text))).squeeze(3)
        conved_4 = self.dropout(F.relu(self.convs[4](text))).squeeze(3)
        conved_5 = self.dropout(F.relu(self.convs[5](text))).squeeze(3)


        # max_pool_n = [batch size, n_filters]
        max_pool_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        max_pool_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        max_pool_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        max_pool_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        max_pool_4 = F.max_pool1d(conved_4, conved_4.shape[2]).squeeze(2)
        max_pool_5 = F.max_pool1d(conved_5, conved_5.shape[2]).squeeze(2)

        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat(max_pool, dim = 1)

        # fc(cat) = [1]
        return self.fc(cat) (output)
        """
        
        # text.shape = [batch_size, height, width]
        
        text = text.unsqueeze(1)

        # text.shape = [batch_size, channels, height, width]
        
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        conved_drop = [self.dropout(conv) for conv in conved]
        
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        max_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_drop]
        
        # max_pool_n = [batch_size, n_filters]
        
        cat = torch.cat(max_pool, dim = 1)
        
        # cat = [batch_size, n_filters * n_filters]
        
        return self.fc(cat)

