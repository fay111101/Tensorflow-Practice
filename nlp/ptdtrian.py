#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-10 上午11:17
@Author  : fay
@Email   : fay625@sina.cn
@File    : ptdtrian.py
@Software: PyCharm
"""
import codecs
import os

print(os.getcwd())
import collections
from operator import itemgetter

MODE = "PTB"  # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。

if MODE == "PTB":  # PTB数据处理
    RAW_DATA = "./data/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB_OUTPUT = "./data/ptb.vocab"  # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = "./data/TED_data/train.txt.zh"
    VOCAB_OUTPUT = "./data/zh.vocab"
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = "./data/TED_data/train.txt.en"
    VOCAB_OUTPUT = "./data/en.vocab"
    VOCAB_SIZE = 10000
counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1
# 按词频顺序对单词进行排序。
sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

if MODE == "PTB":
    # 稍后我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表。
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    # 在9.3.2小节处理机器翻译数据时，除了"<eos>"以外，还需要将"<unk>"和句子起始符
    # "<sos>"加入词汇表，并从词汇表中删除低频词汇。
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")

# 按词频顺序对单词进行排序。
sorted_word_to_cnt = sorted(
    counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# MODE = 'PTB_TRAIN'
# MODE = 'PTB_VALID'
MODE = 'PTB_TEST'

if MODE == "PTB_TRAIN":  # PTB训练数据
    RAW_DATA = "./data/PTB_data/ptb.train.txt"  # 训练集数据文件
    VOCAB = "./data/ptb.vocab"  # 词汇表文件
    OUTPUT_DATA = "./data/ptb.train"  # 将单词替换为单词编号后的输出文件
elif MODE == "PTB_VALID":  # PTB验证数据
    RAW_DATA = "./data/PTB_data/ptb.valid.txt"
    VOCAB = "./data/ptb.vocab"
    OUTPUT_DATA = "./data/ptb.valid"
elif MODE == "PTB_TEST":  # PTB测试数据
    RAW_DATA = "./data/PTB_data/ptb.test.txt"
    VOCAB = "./data/ptb.vocab"
    OUTPUT_DATA = "./data/ptb.test"
elif MODE == "TRANSLATE_ZH":  # 中文翻译数据
    RAW_DATA = "./data/TED_data/train.txt.zh"
    VOCAB = "./data/zh.vocab"
    OUTPUT_DATA = "./data/train.zh"
elif MODE == "TRANSLATE_EN":  # 英文翻译数据
    RAW_DATA = "./data/TED_data/train.txt.en"
    VOCAB = "./data/en.vocab"
    OUTPUT_DATA = "./data/train.en"

with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


# 如果出现了不在词汇表内的低频词，则替换为"unk"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()
