from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import softmax
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date


class YelpDataSet(object):
    def __init__(self):
        self.business = None
        self.reviews = None
        self.review_total_counts = 0
        self.business_total_counts = 0

    def loadBusinessDataSet(self, file_path = './Data/yelp_dataset/yelp_academic_dataset_business.json'):

        # Read the 'business.json' file into a DataFrame
        chunks = pd.read_json(file_path, lines=True, encoding='utf-8', chunksize=10000)
        self.business = pd.DataFrame()
        for chunk in chunks:
            self.business = pd.concat([self.business, chunk])
        print(self.business.columns)
        self.business_total_counts = self.business.shape[0]

    def loadReviews(self, review_file_path = './Data/yelp_dataset/yelp_academic_dataset_review.json'):


        # Read the 'reviews.json' file into a DataFrame
        chunks = pd.read_json(review_file_path, lines=True, encoding='utf-8', chunksize=10000)
        self.reviews = pd.DataFrame()
        for chunk in chunks:
            self.reviews = pd.concat([self.reviews, chunk])
        self.review_total_counts = self.reviews.shape[0]

    def cleanBusiness(self):
        # drop rows that don't have categories
        self.business = self.business.dropna(subset=['categories'])

        # filter by city (Philadelphia)
        philadelphia_businesses = self.business[self.business['city'] == 'Philadelphia']

        # check for 'coffee' in the 'categories' column to identify coffee shops
        self.business = philadelphia_businesses[philadelphia_businesses['categories'].str.startswith('Coffee & Tea')]

    def cleanReview(self):
        # define date threshold (5 years from today)
        date_threshold = (datetime.now() - timedelta(days=5 * 365)).date()

        # Ensure that the 'date' column in Reviews is a Pandas datetime object
        # Convert it to date
        self.reviews['date'] = self.reviews['date'].dt.date
        # Filter reviews dataset by date
        recent_reviews = self.reviews[self.reviews['date'] >= date_threshold]
        # Group and count reviews by business_id
        review_counts = recent_reviews['business_id'].value_counts().reset_index()
        review_counts.columns = ['business_id', 'review_count']
        # Filter philly coffee shops with at least 25 recent reviews
        philly_25_reviews = self.business[self.business['business_id'].
            isin(review_counts[review_counts['review_count'] >= 25]['business_id'])]

        # Inner join to get reviews for businesses that are in both philly_25_reviews and recent_reviews
        filtered_philly_reviews = pd.merge(philly_25_reviews, recent_reviews, on='business_id', how='inner')

        # Group the DataFrame by 'business_id' and count the number of rows (reviews) for each group
        filtered_philly_reviews['recent_review_count'] = filtered_philly_reviews.groupby('business_id')[
            'business_id'].transform('count')
        # replace 'review_count' to 'recent_review_count'

        filtered_philly_reviews['review_count'] = filtered_philly_reviews['recent_review_count']

        filtered_philly_reviews = filtered_philly_reviews.iloc[:, :-1]
        filtered_philly_reviews.sort_values('date')
        filtered_philly_reviews.sort_values('review_count', ascending=False)
        self.reviews = filtered_philly_reviews
        self.reviews.rename(columns={"text":"review_text", "date":"review_date"}, inplace=True)


# The data file can be downloaded from https://www.yelp.com/dataset/download, make sure to unzip it to Data folder.
# Otherwise, need to specify the file path for the loadxxx functions
if __name__ == "__main__":
    yd = YelpDataSet()
    yd.loadBusinessDataSet()
    print(yd.business.head(5))
    yd.loadReviews()
    print(yd.reviews.columns)
    print("total business:", yd.business_total_counts)
    print("total reviews:", yd.review_total_counts)
    yd.cleanBusiness()
    yd.cleanReview()
    yd.reviews.to_csv('filtered_philly_reviews.csv')
    yd.business.to_csv('business_info.csv')

