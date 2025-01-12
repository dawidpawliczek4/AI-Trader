from typing import List
from nlp.data_preprocessing import punctuation_removal, stem
import nltk
from nltk.data import find
import spacy 
import numpy as np

class Tokenizer:
    '''
        class for tokenize text 
    '''
    words = {}
    tokens = {}
    iter = 2
    
    # 0 -> placecholder
    # 1 -> unknown word
    # _ -> normal word
    
    def __init__(self):
        try:
            find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
            
        try: 
            self.nlp = spacy.load('en_core_web_lg')
        except:
            from spacy.cli import download 
            download('en_core_web_lg') 
            self.nlp = spacy.load('en_core_web_lg')
     
    def fit_on_sequences(self, sequences: List[str]):
        '''
            before making tokens you have to use this function
            
            intput:
                sequences: list(str)    
            output:
                void 
        '''
        
        # first we have to count all words
        for sequence in sequences:
            for word in sequence.split():
                self.words[word] = self.words.get(word, 0) + 1
                
        # sort words
        sorted_words = sorted(self.words.items(), key=lambda x: x[1], reverse=True)
        
        # now we can make tokens
        for word, _ in sorted_words:
            self.tokens[word] = self.iter 
            self.iter += 1
        
    def ibe_text(self, text: str, max_token_no: int, sequence_len: int) -> List[int]:
        '''
            function for tokenizing text, eg "I love reading books" -> [4, 5, 7, 8, 0, 0, 0, 0, 0]
            input: 
                text: string, text that you want to tokenize 
                max_token_no: int, how big can be the biggest token
                sequence_len: int, how long sequence you want 
            return: 
                list of tokens (less than max_token_no) of sieze sequence_len 
        '''
        
        ans = [0] * sequence_len
        seq = text.split()
        for i in range(len(seq)):
            word = seq[i]
            ans[i] = self.tokens.get(word, 1) if self.tokens.get(word, 1) < max_token_no else 1
        
        return ans
    
    def bow_text(self, text: str, max_token_no: int) -> List[int]:
        '''
            function for createing bag of words from a given text
            input:
                text: str 
                max_token_no: int, size of bow
            return:
                List[int], bow
        '''
        ans = [0] * max_token_no
        seq = text.split()
        for word in seq:
            token = self.tokens.get(word, 1) if self.tokens.get(word, 1) < max_token_no else 1
            ans[token] += 1
        
        return ans
    
    def word2vec_text(self, text: str) -> np.array:
        '''
            function for converting text into vector
            input: 
                text: str
            return: 
                np.array 300x1
        '''
        
        return self.nlp(text).vector
        
        
            
    def clean_text(self, text: str) -> str:
        '''
            function for cleaning text 
            input:
                text: str
            return 
                str
        '''
        
        text = punctuation_removal(text)
        text = text.replace('amp', '').lower()
        text = stem(text)
        return text
    