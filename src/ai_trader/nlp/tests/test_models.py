def test_sentiment_analyzer(sentiment_analyzer, custom_X,
                            custom_y, sample_sequences):
    sentiment_analyzer.fit(custom_X, custom_y)
    sentiment_analyzer.evaluate()
    y = sentiment_analyzer.predict(sample_sequences)
    assert isinstance(y, list)
    assert len(y) == len(sample_sequences)