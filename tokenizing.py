import pandas as pd
from konlpy.tag import Mecab
from tokenizers import BertWordPieceTokenizer

# 데이터 불러오기
data = pd.read_csv('data/corpus_shuf.tsv', sep='\t', header=None, names=['kr', 'en'], on_bad_lines='warn')

# 영어 번역 결과가 없는 7개 문장 제거
data = data.dropna()

# 한국어, 영어 문장 분리 및 저장
with open('./data/corpus_kr.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(data['kr']))
with open('./data/corpus_en.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(data['en']))


# Mecab
def tokenize_mecab(sentences):
    tokenizer = Mecab(dicpath='c:/mecab/mecab-ko-dic')
    tokens_kr_mc = [tokenizer.morphs(s) for s in sentences]
    return tokens_kr_mc


# BertWordPieceTokenizer
file_path_kr = './data/corpus_kr.txt'
file_path_en = './data/corpus_en.txt'
vocab_size = 30000
limit_alphabet = 6000
min_frequency = 5

# 한국어 학습
tokenizer_wp_kr = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
tokenizer_wp_kr.train(files=file_path_kr,
                      vocab_size=vocab_size,
                      limit_alphabet=limit_alphabet,
                      min_frequency=min_frequency,
                      wordpieces_prefix='##'
                      )

# 영어 학습
tokenizer_wp_en = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
tokenizer_wp_en.train(files=file_path_en,
                      vocab_size=vocab_size,
                      limit_alphabet=limit_alphabet,
                      min_frequency=min_frequency,
                      wordpieces_prefix='##'
                      )

# 단어 사전 저장
tokenizer_wp_kr.save_model('.', 'kr')
tokenizer_wp_en.save_model('.', 'en')
