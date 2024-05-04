
from gensim.models import KeyedVectors

# 加载Word2Vec模型
model = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)

# 输出词向量的维度
print("词向量维度:", model.vector_size)
print("词数量:", len(model))

# 获取某个词的词向量
word = "足球"
if word in model:
    vector = model[word]
    print(f"{word}的词向量:", vector)
else:
    print(f"{word}不在词汇表中。")

# 找到与某个词最相似的词
similar_words = model.most_similar("足球", topn=5)
print("与'足球'最相似的词:")
for word, similarity in similar_words:
    print(f"词: {word}, 相似度: {similarity}")

model[word]