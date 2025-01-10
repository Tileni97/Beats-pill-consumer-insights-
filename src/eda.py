import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Dict, List, Tuple

class ExploratoryAnalysis:
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame"""
        self.df = df
        self.setup_visualization_style()

    def setup_visualization_style(self):
        """Configure visualization settings"""
        plt.style.use('seaborn')
        sns.set_palette("deep")
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        self.color_palette = {
            'Beats Pill': '#FF0000',      # Red for Beats
            'positive': '#2ECC71',        # Green for positive
            'neutral': '#95A5A6',         # Grey for neutral
            'negative': '#E74C3C',        # Red for negative
            'JBL': '#0000FF',            # Blue for JBL
            'Bose': '#FFA500'            # Orange for Bose
        }

    def analyze_rating_distribution(self) -> Dict:
        """Analyze rating distribution by product"""
        # Calculate statistics
        rating_stats = self.df.groupby('product_name')['rating'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2).sort_values('mean', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='product_name', y='rating')
        plt.title('Rating Distribution by Product', fontsize=14, pad=20)
        plt.xlabel('Product', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return rating_stats.to_dict()

    def analyze_temporal_patterns(self) -> pd.DataFrame:
        """Analyze rating trends over time"""
        # Calculate monthly averages
        monthly_ratings = self.df.groupby([
            pd.Grouper(key='review_date', freq='M'),
            'product_name'
        ])['rating'].mean().reset_index()
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        for product in self.df['product_name'].unique():
            product_data = monthly_ratings[monthly_ratings['product_name'] == product]
            plt.plot(product_data['review_date'], 
                    product_data['rating'],
                    label=product,
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6)
        
        plt.title('Average Rating Trends Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        return monthly_ratings

    def analyze_review_engagement(self) -> Dict:
        """Analyze patterns in review engagement"""
        # Calculate correlation
        correlation = self.df['rating'].corr(self.df['helpful_votes'])
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, 
                       x='rating', 
                       y='helpful_votes', 
                       hue='product_name', 
                       alpha=0.6)
        plt.title('Rating vs Review Helpfulness', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Helpful Votes', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate engagement statistics
        engagement_stats = self.df.groupby('rating').agg({
            'helpful_votes': ['mean', 'count', 'sum']
        }).round(2)
        
        return {
            'correlation': correlation,
            'engagement_stats': engagement_stats.to_dict()
        }

    def analyze_verification_impact(self) -> Dict:
        """Analyze impact of verified purchases"""
        # Calculate verification statistics
        verification_stats = self.df.groupby(
            ['product_name', 'verified_purchase']
        )['rating'].agg(['mean', 'count', 'std']).round(2)
        
        # Calculate verification percentages
        verification_percentages = (self.df.groupby('product_name')['verified_purchase']
                                  .value_counts(normalize=True)
                                  .mul(100)
                                  .round(1)
                                  .unstack())
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=self.df, 
                   x='verified_purchase', 
                   y='rating', 
                   hue='product_name')
        plt.title('Rating Distribution: Verified vs Unverified Purchases', 
                 fontsize=14, pad=20)
        plt.xlabel('Verified Purchase Status', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'verification_stats': verification_stats.to_dict(),
            'verification_percentages': verification_percentages.to_dict()
        }

    def analyze_seasonal_patterns(self) -> Dict:
        """Analyze seasonal patterns in reviews"""
        # Add season column if not present
        if 'season' not in self.df.columns:
            self.df['season'] = pd.cut(
                self.df['review_date'].dt.month,
                bins=[0, 3, 6, 9, 12],
                labels=['Winter', 'Spring', 'Summer', 'Fall']
            )
        
        # Calculate seasonal statistics
        seasonal_stats = self.df.groupby(['season', 'product_name'])['rating'].agg([
            'mean', 'count', 'std'
        ]).round(2)
        
        # Calculate review volume by season
        seasonal_volume = self.df.groupby(['season', 'product_name']).size().unstack()
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Rating Distribution by Season
        sns.boxplot(data=self.df, 
                   x='season', 
                   y='rating', 
                   hue='product_name',
                   ax=ax1)
        ax1.set_title('Rating Distribution by Season', fontsize=14, pad=20)
        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Rating', fontsize=12)
        ax1.legend().remove()
        
        # Review Volume by Season
        seasonal_volume.plot(kind='bar', ax=ax2)
        ax2.set_title('Review Volume by Season', fontsize=14, pad=20)
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Number of Reviews', fontsize=12)
        ax2.legend(title='Product', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.show()
        
        return {
            'seasonal_stats': seasonal_stats.to_dict(),
            'seasonal_volume': seasonal_volume.to_dict()
        }

    def generate_summary_report(self) -> Dict:
        """Generate a comprehensive summary report"""
        report = {
            'rating_distribution': self.analyze_rating_distribution(),
            'engagement_metrics': self.analyze_review_engagement(),
            'verification_impact': self.analyze_verification_impact(),
            'seasonal_patterns': self.analyze_seasonal_patterns()
        }
        return report