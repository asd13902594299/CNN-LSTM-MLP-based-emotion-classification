import torch
from gensim.models import KeyedVectors
from Model.CNN import CNN
from utils import *
from preprocess import get_data_loader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2vec = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)

EMBEDDING_DIM = word2vec.vector_size
FILTER_SIZES = [3,4,5,6]
N_FILTERS = len(FILTER_SIZES)
OUTPUT_DIM = 1
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
    model = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device=device)
    model.load_state_dict(torch.load('./Torch/CNN-model.pt'))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    
    test_loader, train_loader, validation_loader = get_data_loader(batch_size=16, validation=True)
    test_loss, test_acc, test_f_score = evaluate(model, test_loader, criterion)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | F-score: {test_f_score:.2f}')
    
    while False:
        s = input()
        print(predict(model, s).item())
