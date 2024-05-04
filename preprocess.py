from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors
import torch
import numpy as np
import os

MAX_SENTENCE_LEN = 700

def build_label_vectors(type_: str, word2vec):
    labels_path = f"./Torch/{type_}_labels.pt"
    sentence_path = f"./Torch/{type_}_sentences.pt"
    sentence_len_path = f"./Torch/{type_}_sentence_len.pt"
    if not os.path.exists(labels_path) or not os.path.exists(sentence_path):
        labels = []
        vectors = []
        slen = []
        with open(f"Dataset/{type_}.txt", "r") as f:
            for line in f.readlines():
                sentence = list(line[2:-1].split(" "))
                slen.append(len(sentence))
                for idx, s in enumerate(sentence):
                    if s not in word2vec:
                        sentence[idx] = "<u>"
                sentence.extend(['<p>' for _ in range(MAX_SENTENCE_LEN-len(sentence))])
                vectors.append([word2vec[s] for s in sentence])
                labels.append(int(line[0]))
                
        torch.save(torch.tensor(np.array(labels)), labels_path)
        torch.save(torch.tensor(np.array(vectors)), sentence_path)
        torch.save(torch.tensor(np.array(slen)), sentence_len_path)


def get_data_loader(batch_size = 1, validation = False, shuffle = True) -> tuple[DataLoader]:
    Testset = TensorDataset(
        torch.load("./Torch/test_labels.pt").cuda(), 
        torch.load("./Torch/test_sentences.pt").cuda(),
        torch.load("./Torch/test_sentence_len.pt").cuda(),
    )
    Traintest = TensorDataset(
        torch.load("./Torch/train_labels.pt").cuda(), 
        torch.load("./Torch/train_sentences.pt").cuda(),
        torch.load("./Torch/train_sentence_len.pt").cuda(),
    )
    if (validation):
        Validationset = TensorDataset(
            torch.load("./Torch/validation_labels.pt").cuda(), 
            torch.load("./Torch/validation_sentences.pt").cuda(),
            torch.load("./Torch/validation_sentence_len.pt").cuda(),
        )
        return (
            DataLoader(dataset=Testset, batch_size=batch_size, shuffle=shuffle), 
            DataLoader(dataset=Traintest, batch_size=batch_size, shuffle=shuffle), 
            DataLoader(dataset=Validationset, batch_size=batch_size, shuffle=shuffle)
        )
    return (
        DataLoader(dataset=Testset, batch_size=batch_size, shuffle=shuffle), 
        DataLoader(dataset=Traintest, batch_size=batch_size, shuffle=shuffle), 
    )


if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)
    word2vec["<u>"] = np.array([0]*50)
    word2vec["<p>"] = np.array([0]*50)

    build_label_vectors("test", word2vec)
    build_label_vectors("train", word2vec)
    build_label_vectors("validation", word2vec)
        
    # d = TensorDataset(torch.load("./Torch/train_labels.pt"), torch.load("./Torch/train_sentences.pt"))
    # l = DataLoader(dataset=d, batch_size=4, shuffle=False)
    # for batch in l:
    #     print(batch[0], batch[1])