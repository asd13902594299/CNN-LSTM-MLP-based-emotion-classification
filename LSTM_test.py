import torch
from gensim.models import KeyedVectors
from Model.LSTM import LSTM
from utils import *
from preprocess import get_data_loader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2vec = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)

EMBEDDING_DIM = word2vec.vector_size
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
MAX_SENTENCE_LEN = 700


def predict(model, sentence):
    sentence = list(sentence.split(" "))
    for idx, s in enumerate(sentence):
        if s not in word2vec:
            sentence[idx] = "<u>"
    sentence.extend(['<p>' for _ in range(MAX_SENTENCE_LEN-len(sentence))])
    return torch.sigmoid(model(torch.tensor(np.array([word2vec[s] for s in sentence])).unsqueeze(0).cuda()))


if __name__ == '__main__':
    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, MAX_SENTENCE_LEN).to(device=device)
    model.load_state_dict(torch.load('./Torch/LSTM-model.pt'))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    test_loader, train_loader, validation_loader = get_data_loader(batch_size=128, validation=True)
    test_loss, test_acc, test_f_score = evaluate(model, test_loader, criterion)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test F-score: {test_f_score}')
    
    while False:
        s = input()
        print(predict(model, s).item())
