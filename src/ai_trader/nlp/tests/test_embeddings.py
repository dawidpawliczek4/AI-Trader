import numpy as np

def test_ibe_text(tokenizer, sample_text, sample_sequences):
    tokenizer.fit_on_sequences(sample_sequences)
    result = tokenizer.ibe_text(sample_text, max_token_no=5000, sequence_len=20)
    
    assert isinstance(result, list)
    assert len(result) == 20
    assert all(isinstance(token, int) for token in result)
    
def test_bow_text(tokenizer, sample_text, sample_sequences):
    tokenizer.fit_on_sequences(sample_sequences)
    result = tokenizer.bow_text(sample_text, max_token_no=5000)
    
    assert isinstance(result, list)
    assert len(result) == 5000
    assert all(isinstance(count, int) for count in result) 
    
def test_word2vec_text(tokenizer, sample_text, sample_sequences):
    tokenizer.fit_on_sequences(sample_sequences)
    result = tokenizer.word2vec_text(sample_text)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (300, )