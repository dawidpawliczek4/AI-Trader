import argparse
from ai_trader.nlp.sentiment_analyzer import Sentiment_analyzer

def train():
    parser = argparse.ArgumentParser(description='Train sentiment analyzer')
    parser.add_argument('--word_transformation', type=str, default='bow', 
                       help='Word transformation method')
    parser.add_argument('--model', type=str, default='nn',
                       help='Model type')
    parser.add_argument('--load_path', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--save_path', type=str, default='sentiment_model.pth',
                       help='Path to save trained model')
    args = parser.parse_args()
    sentiment_analyzer = Sentiment_analyzer(
        word_transformation=args.word_transformation,
        model=args.model,
        load_path=args.load_path
    )
    sentiment_analyzer.fit(save_path=args.save_path)

if __name__ == "__main__":
    train()