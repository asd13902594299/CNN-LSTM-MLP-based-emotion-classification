import torch
from gensim.models import KeyedVectors
import random

def binary_accuracy_F_score(preds, y):
    """
    Returns accuracy and F-score per batch
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    recall = ((rounded_preds == 0)*(y == 0).float()).sum() / (y == 0).float().sum()
    precision = ((rounded_preds == 0)*(y == 0).float()).sum() / (rounded_preds == 0).float().sum()
    acc = correct.sum() / len(correct)
    return acc, 2/((1/precision)+(1/recall))


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_f_score = 0 
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        sentence_len = batch[2]
        
        sentence = batch[1]
        
        label = batch[0]
                            
        predictions = model(sentence, sentence_len).squeeze(1)
        
        loss = criterion(predictions, label.float())
        
        acc, f_score = binary_accuracy_F_score(predictions, label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f_score += f_score
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f_score / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_f_score = 0 
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            
            sentence_len = batch[2]
        
            sentence = batch[1]
            
            label = batch[0]
                                
            predictions = model(sentence, sentence_len).squeeze(1)
            
            loss = criterion(predictions, label.float())
                        
            acc, f_score = binary_accuracy_F_score(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f_score += f_score
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f_score / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

