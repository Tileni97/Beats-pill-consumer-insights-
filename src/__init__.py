from src.data_cleaning import DataCleaner
from src.eda import ExploratoryAnalysis
from src.sentiment_analysis import SentimentAnalyzer
import pandas as pd

def main():
    # Initialize classes
    cleaner = DataCleaner()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Load and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv('data/merged_reviews.csv')
    df_cleaned = cleaner.clean_data(df)
    df_cleaned = cleaner.map_product_names(df_cleaned)
    
    # Perform sentiment analysis
    print("\nPerforming sentiment analysis...")
    df_with_sentiment = sentiment_analyzer.process_reviews(df_cleaned)
    
    # Generate sentiment report
    sentiment_report = sentiment_analyzer.generate_sentiment_report(df_with_sentiment)
    
    # Initialize exploratory analysis
    explorer = ExploratoryAnalysis(df_with_sentiment)
    
    # Generate EDA report
    print("\nGenerating exploratory analysis...")
    eda_report = explorer.generate_summary_report()
    
    # Save processed data
    df_with_sentiment.to_csv('data/processed_reviews.csv', index=False)
    print("\nAnalysis complete! Processed data saved to 'processed_reviews.csv'")
    
    return df_with_sentiment, sentiment_report, eda_report

if __name__ == "__main__":
    df, sentiment_report, eda_report = main()