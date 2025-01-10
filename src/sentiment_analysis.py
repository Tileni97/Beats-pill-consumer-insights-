import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Tuple

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis tools"""
        # Initialize stop words
        self.stop_words = set(stopwords.words('english'))
        self.product_terms = {
            'speaker', 'speakers', 'bluetooth', 'beats', 'pill',
            'device', 'product', 'bought', 'purchase', 'buying',
            'amazon', 'ordered', 'received'
        }
        self.stop_words.update(self.product_terms)

    def clean_text(self, text: str) -> str:
        """Clean and standardize review text"""
        if not isinstance(text, str):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into words
        words = word_tokenize(text)
        
        # Remove stop words and short words
        cleaned_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(cleaned_words)

    def analyze_sentiment(self, text: str) -> Dict:
        """Calculate sentiment scores for text"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def categorize_sentiment(self, polarity: float) -> str:
        """Categorize sentiment based on polarity score"""
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        return 'Neutral'

    def process_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all reviews in the DataFrame"""
        print("Processing reviews...")
        
        # Clean text
        df['processed_review'] = df['review_content'].apply(self.clean_text)
        print("✓ Reviews cleaned")
        
        # Calculate sentiment scores
        sentiment_scores = df['processed_review'].apply(self.analyze_sentiment)
        df['polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        df['subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        print("✓ Sentiment scores calculated")
        
        # Categorize sentiments
        df['sentiment_category'] = df['polarity'].apply(self.categorize_sentiment)
        print("✓ Sentiments categorized")
        
        return df

    def generate_wordcloud(self, df: pd.DataFrame, sentiment: str = None) -> None:
        """Generate word cloud for reviews"""
        # Filter by sentiment if specified
        if sentiment:
            text = ' '.join(df[df['sentiment_category'] == sentiment]['processed_review'])
            title = f'Word Cloud - {sentiment} Reviews'
        else:
            text = ' '.join(df['processed_review'])
            title = 'Word Cloud - All Reviews'
        
        # Create and generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()

    def analyze_ngrams(self, df: pd.DataFrame, n: int = 2, top_n: int = 10) -> Dict:
        """Analyze n-grams in reviews"""
        ngram_results = {}
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            # Filter reviews by sentiment
            reviews = df[df['sentiment_category'] == sentiment]['processed_review']
            
            # Generate n-grams
            all_ngrams = []
            for review in reviews:
                tokens = word_tokenize(review)
                all_ngrams.extend(list(ngrams(tokens, n)))
            
            # Get top N most common n-grams
            ngram_freq = Counter(all_ngrams).most_common(top_n)
            ngram_results[sentiment] = ngram_freq
        
        return ngram_results

    def analyze_sentiment_distribution(self, df: pd.DataFrame) -> None:
        """Analyze and visualize sentiment distribution"""
        plt.figure(figsize=(15, 5))
        
        # Sentiment Polarity Distribution
        plt.subplot(131)
        sns.histplot(data=df, x='polarity', bins=30)
        plt.title('Distribution of Sentiment Polarity')
        plt.xlabel('Polarity Score')
        
        # Sentiment Category Distribution
        plt.subplot(132)
        counts = df['sentiment_category'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        
        # Rating by Sentiment
        plt.subplot(133)
        sns.boxplot(data=df, x='sentiment_category', y='rating')
        plt.title('Rating Distribution by Sentiment')
        
        plt.tight_layout()
        plt.show()

    def get_common_terms(self, df: pd.DataFrame, sentiment_type: str, n: int = 10) -> List[Tuple]:
        """Extract most common terms for a sentiment category"""
        # Filter reviews
        category_reviews = df[
            (df['sentiment_category'] == sentiment_type) & 
            (df['product_name'] == 'Beats Pill')
        ]
        
        # Combine all processed reviews
        all_words = ' '.join(category_reviews['processed_review']).split()
        
        return Counter(all_words).most_common(n)

    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive sentiment analysis report"""
        report = {
            'overall_metrics': {
                'average_polarity': df['polarity'].mean(),
                'average_subjectivity': df['subjectivity'].mean(),
                'sentiment_distribution': df['sentiment_category'].value_counts().to_dict()
            },
            'by_product': df.groupby('product_name').agg({
                'polarity': 'mean',
                'subjectivity': 'mean'
            }).round(3).to_dict(),
            'common_terms': {
                sentiment: self.get_common_terms(df, sentiment)
                for sentiment in ['Positive', 'Neutral', 'Negative']
            }
        }
        
        return report