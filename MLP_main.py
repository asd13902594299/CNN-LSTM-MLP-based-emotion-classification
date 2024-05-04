import torch
import time
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from Model.MLP import MLP
from utils import *
from preprocess import get_data_loader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2vec = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)
word2vec["<u>"] = np.array([0]*50)
word2vec["<p>"] = np.array([0]*50)

EMBEDDING_DIM = word2vec.vector_size
OUTPUT_DIM = 1
DROPOUT = 0.5
MAX_SENTENCE_LEN = 700
TRUNCATE_LEN = 75
HIDDEN_DIM = int((TRUNCATE_LEN*EMBEDDING_DIM))

N_EPOCHS = 20

if __name__ == '__main__':
    model = MLP(MAX_SENTENCE_LEN, EMBEDDING_DIM, TRUNCATE_LEN, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    test_loader, train_loader, validation_loader = get_data_loader(batch_size=128, validation=True)

    best_valid_loss = 0 # 0 for no updating model 
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc, train_f_score = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_f_score = evaluate(model, validation_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './Torch/MLP-model.pt')
        
        if epoch < 10:
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | F-score: {train_f_score:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | F-score: {valid_f_score:.2f}')
    
    print('after training...')
    print(f'Epoch: {N_EPOCHS:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | F-score: {train_f_score:.2f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | F-score: {valid_f_score:.2f}')
    test_loss, test_acc, test_f_score = evaluate(model, test_loader, criterion)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}% | F-score: {test_f_score:.2f}')
    print(train_losses)
    print(valid_losses)
    print(train_accs)
    print(valid_accs)
