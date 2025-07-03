import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")


class AnalystSentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model"""
        print("Loading sentiment analysis model...")

        # Using a robust financial sentiment model
        # Alternative: "nlptown/bert-base-multilingual-uncased-sentiment" for general sentiment
        model_name = "ProsusAI/finbert"

        try:
            # Try to load FinBERT (specifically trained on financial text)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ FinBERT model loaded successfully")
        except:
            # Fallback to general sentiment model
            print("FinBERT not available, using general sentiment model...")
            self.classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ General sentiment model loaded successfully")

    def analyze_sentiment(self, text):
        """Analyze sentiment of a single recommendation"""
        try:
            result = self.classifier(text)
            return result[0]
        except Exception as e:
            return {"label": "ERROR", "score": 0.0, "error": str(e)}

    def interpret_sentiment(self, result):
        """Interpret the sentiment result with more context"""
        if "error" in result:
            return f"Error analyzing sentiment: {result['error']}"

        label = result['label'].upper()
        confidence = result['score']

        # Map different model outputs to consistent labels
        if label in ['POSITIVE', 'POS', 'BULLISH']:
            sentiment = "POSITIVE"
            interpretation = "Bullish/Favorable"
        elif label in ['NEGATIVE', 'NEG', 'BEARISH']:
            sentiment = "NEGATIVE"
            interpretation = "Bearish/Unfavorable"
        elif label in ['NEUTRAL', 'NEU']:
            sentiment = "NEUTRAL"
            interpretation = "Neutral/Mixed"
        else:
            sentiment = label
            interpretation = "Unknown"

        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"

        return {
            "sentiment": sentiment,
            "interpretation": interpretation,
            "confidence": confidence,
            "confidence_level": confidence_level
        }

    def batch_analyze(self, recommendations):
        """Analyze multiple recommendations at once"""
        results = []
        for i, rec in enumerate(recommendations, 1):
            print(f"\nAnalyzing recommendation {i}...")
            result = self.analyze_sentiment(rec)
            interpreted = self.interpret_sentiment(result)
            results.append({
                "recommendation": rec,
                "analysis": interpreted
            })
        return results


def main():
    """Main function to run the sentiment analyzer"""
    print("=" * 60)
    print("ANALYST RECOMMENDATION SENTIMENT ANALYZER")
    print("=" * 60)

    # Initialize analyzer
    analyzer = AnalystSentimentAnalyzer()

    print("\nOptions:")
    print("1. Analyze single recommendation")
    print("2. Analyze multiple recommendations")
    print("3. Quit")

    while True:
        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            print("\n" + "─" * 40)
            recommendation = input("Enter analyst recommendation: ").strip()

            if recommendation:
                print("\nAnalyzing...")
                result = analyzer.analyze_sentiment(recommendation)
                analysis = analyzer.interpret_sentiment(result)

                print(f"\n📊 SENTIMENT ANALYSIS RESULTS")
                print(f"─" * 30)
                print(f"Text: {recommendation}")
                print(f"Sentiment: {analysis['sentiment']}")
                print(f"Interpretation: {analysis['interpretation']}")
                print(f"Confidence: {analysis['confidence']:.3f} ({analysis['confidence_level']})")

                # Add investment guidance
                if analysis['sentiment'] == 'POSITIVE':
                    print("💡 Suggestion: Consider as a potential buy/hold opportunity")
                elif analysis['sentiment'] == 'NEGATIVE':
                    print("⚠️  Suggestion: Exercise caution, consider sell/avoid")
                else:
                    print("📊 Suggestion: Mixed signals, conduct further analysis")

        elif choice == "2":
            print("\n" + "─" * 40)
            print("Enter multiple recommendations (press Enter twice when done):")
            recommendations = []

            while True:
                rec = input("Recommendation: ").strip()
                if not rec:
                    break
                recommendations.append(rec)

            if recommendations:
                results = analyzer.batch_analyze(recommendations)

                print(f"\n📊 BATCH ANALYSIS RESULTS")
                print("═" * 50)

                for i, result in enumerate(results, 1):
                    analysis = result['analysis']
                    print(f"\n{i}. {result['recommendation'][:60]}...")
                    print(f"   Sentiment: {analysis['sentiment']} ({analysis['confidence']:.3f})")
                    print(f"   Interpretation: {analysis['interpretation']}")

                # Summary statistics
                sentiments = [r['analysis']['sentiment'] for r in results]
                pos_count = sentiments.count('POSITIVE')
                neg_count = sentiments.count('NEGATIVE')
                neu_count = sentiments.count('NEUTRAL')

                print(f"\n📈 SUMMARY:")
                print(f"   Positive: {pos_count}/{len(results)}")
                print(f"   Negative: {neg_count}/{len(results)}")
                print(f"   Neutral: {neu_count}/{len(results)}")

                if pos_count > neg_count:
                    print("   📊 Overall sentiment: BULLISH")
                elif neg_count > pos_count:
                    print("   📊 Overall sentiment: BEARISH")
                else:
                    print("   📊 Overall sentiment: MIXED")

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()