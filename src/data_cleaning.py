import pandas as pd
import numpy as np
import re
from typing import Dict, List

class DataCleaner:
    def __init__(self):
        """Initialize data cleaning tools"""
        self.product_names = {
            "B0D4SX9RC6": "Beats Pill",
            "B07DD3W154": "MEGABOOM 3",
            "B08MZZTH1N": "Tribit",
            "B088KRKFJ3": "Marshall Stockwell II",
            "B08VKXP1VY": "Bose SoundLink Revolve",
            "B07P39MLKH": "Soundcore Motion+",
            "B07YFXRNHF": "Monster S310",
            "B0D95YFJXS": "ZEALOT",
            "B08X4XBB26": "JBL CHARGE 5",
            "B09XXW54QG": "Marshall Emberton",
            "B0CY6S748H": "Sonos Roam 2",
            "B09GK5JMHK": "JBL Flip 6",
            "B0B43Y8GHZ": "Sony SRS-XG300",
            "B085R7TSN6": "Bang & Olufsen Beosound A1",
            "B09SNYHYV7": "Tronsmart"
        }

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main data cleaning function"""
        print("Starting data cleaning process...")
        df_cleaned = df.copy()
        
        # Rename columns
        column_mapping = {
            'title': 'review_title',
            'author': 'reviewer_name',
            'content': 'review_content',
            'timestamp': 'review_date',
            'profile_id': 'reviewer_profile_id',
            'is_verified': 'verified_purchase',
            'helpful_count': 'helpful_votes',
            'product_attributes': 'product_variant'
        }
        df_cleaned = df_cleaned.rename(columns=column_mapping)
        print("✓ Columns renamed successfully")
        
        # Handle missing values
        df_cleaned['review_content'] = df_cleaned['review_content'].fillna('')
        df_cleaned['product_variant'] = df_cleaned['product_variant'].fillna('Unknown')
        print("✓ Missing values handled")
        
        # Remove duplicates
        initial_length = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_length - len(df_cleaned)
        print(f"✓ Removed {duplicates_removed} duplicate entries")
        
        # Drop unnecessary columns
        columns_to_drop = ['review_id', 'reviewer_name', 'reviewer_profile_id']
        df_cleaned = df_cleaned.drop(columns=columns_to_drop)
        print("✓ Unnecessary columns removed")
        
        # Clean dates
        df_cleaned = self.clean_dates(df_cleaned)
        print("✓ Dates cleaned")
        
        # Handle outliers in numeric columns
        numeric_columns = ['rating', 'helpful_votes']
        for column in numeric_columns:
            df_cleaned = self.handle_outliers(df_cleaned, column)
        print("✓ Outliers handled")
        
        return df_cleaned

    def handle_outliers(self, df: pd.DataFrame, column: str, method='winsorize') -> pd.DataFrame:
        """Handle outliers in specified column"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
        print(f"Found {outliers_count} outliers in {column}")
        
        if method == 'winsorize':
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"Outliers winsorized for {column}")
        
        return df

    def map_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map product IDs to readable names"""
        print("\nMapping product IDs to names...")
        df['product_name'] = df['product_id'].map(self.product_names)
        print("✓ Product names mapped successfully")
        return df
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize review dates"""
        def extract_date(date_str):
            match = re.search(r'(\w+ \d+, \d{4})', str(date_str))
            return match.group(1) if match else None

        df['review_date'] = pd.to_datetime(
            df['review_date'].apply(extract_date)
        )
        
        # Add useful date features
        df['review_year'] = df['review_date'].dt.year
        df['review_month'] = df['review_date'].dt.month
        df['review_quarter'] = df['review_date'].dt.quarter
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate a data quality report"""
        report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_products': df['product_name'].nunique(),
            'date_range': {
                'start': df['review_date'].min(),
                'end': df['review_date'].max()
            },
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'verified_purchases': df['verified_purchase'].value_counts().to_dict()
        }
        return report

    def extract_product_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract color and style from product attributes"""
        def extract_color(attr):
            color_match = re.search(r'Color: ([^,]+)', str(attr))
            return color_match.group(1) if color_match else 'Unknown'

        def extract_style(attr):
            style_match = re.search(r'Style: ([^,]+)', str(attr))
            return style_match.group(1) if style_match else 'Unknown'

        df['product_color'] = df['product_variant'].apply(extract_color)
        df['product_style'] = df['product_variant'].apply(extract_style)
        
        return df