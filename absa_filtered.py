from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

class absaFilter(object):
    def __init__(self):
        self.reviews = None
        self.review_rates = None
        self.coffee_keywords = ['coffee', 'espresso', 'latte', 'macchiato',
                                'flat white', 'pour over', 'cappuccino',
                                'cold brew', 'cortado','mocha','americano']
        # Load Aspect-Based Sentiment Analysis model

        large = "yangheng/deberta-v3-large-absa-v1.1"
        base = "yangheng/deberta-v3-base-absa-v1.1"

        self.absa_tokenizer = AutoTokenizer.from_pretrained(base)
        self.absa_model = AutoModelForSequenceClassification.from_pretrained(base)

        # Load a traditional Sentiment Analysis model
        sentiment_model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                                   tokenizer=sentiment_model_path)

    def loadReviews(self, filepath = 'Data/filtered_philly_reviews.csv'):
        self.reviews = pd.read_csv(filepath)
        # remove the first column from df
        self.reviews = self.reviews.iloc[:,1:]


    def testModel(self):
        sentence = 'Such good iced coffee. Tastes like hot coffee tastes but actually is iced coffee. Some iced coffees are more bitter or not enough bitter but this one is the nice amount of bitter. And good ice to coffee ratio. Love the Halloween decorations too. Lovely.'

        # ABSA of coffee_keywords
        aspect = self.coffee_keywords
        inputs = self.absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
        outputs = self.absa_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        probs = probs.detach().numpy()[0]
        print(f"Sentiment of aspect '{aspect}' is:")
        for prob, label in zip(probs, ["negative", "neutral", "positive"]):
            print(f"Label {label}: {prob}")

        sentiment = self.sentiment_model([sentence])[0]
        print(f"Overall sentiment: {sentiment['label']} with score {sentiment['score']}")


    def calPSS(self):
        # Initialize a list to store the pss (positive sentiment score) for each review
        pss_list = []
        positive = []

        # Iterate through each review in the DataFrame and calculate pss
        for index, row in tqdm(self.reviews.iterrows()):
            print("calc:"+ str(index))
            sentence = row['review_text']
            aspect = self.coffee_keywords
            inputs = self.absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
            outputs = self.absa_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pss = probs[0][2].item() # The third value in probs corresponds to the positive sentiment score
            pss_list.append(pss)
            probs_tensor = torch.tensor(probs, requires_grad=True)
            probs = probs.detach().numpy()
            if np.argmax(probs) == 2:
                positive.append(1)
            else:
                positive.append(0)

        # Add the pss as a new column to the DataFrame
        self.reviews['pss'] = pss_list

        # Add the positive as a new column to the DataFrame
        self.reviews['positive'] = positive

        # Group by business and calculate the average pss (Metric 1)
        metric_1 = self.reviews.groupby('business_id')['pss'].mean().reset_index()
        metric_1.columns = ['business_id', 'metric_1']
        # Group by business and calculate the positive sentiment ratio (Metric 2)
        metric_2 = self.reviews.groupby('business_id')['positive'].mean().reset_index()
        metric_2.columns = ['business_id', 'metric_2']

        metrics_stars_x = self.reviews.groupby('business_id')['stars_x'].mean().reset_index()
        metrics_stars_x.columns = ['business_id', 'stars']

        metrics_stars_y = self.reviews.groupby('business_id')['stars_y'].mean().reset_index()
        metrics_stars_y.columns = ['business_id', 'recent_stars']

        # Merge metric_1 and metric_2 with the main DataFrame
        self.reviews = pd.merge(self.reviews, metric_1, on='business_id')
        self.reviews = pd.merge(self.reviews, metric_2, on='business_id')
        self.reviews['composite_score'] = 0.6 * self.reviews['metric_1'] + 0.4 * self.reviews['metric_2']

        self.review_rates = pd.merge(metric_1, metric_2, on='business_id')
        self.review_rates = pd.merge(self.review_rates, metrics_stars_x, on='business_id')
        self.review_rates = pd.merge(self.review_rates, metrics_stars_y, on='business_id')


    def calcTop10(self):
        # Group by business and calculate average composite score
        business_scores = self.reviews.groupby('business_id')['composite_score'].mean().reset_index()

        # Sort by composite score descending
        top_10_coffee_shops = business_scores.sort_values(by='composite_score', ascending=False).head(21)

        # Merge with business information
        top_10_coffee_shops = top_10_coffee_shops.merge(self.reviews[['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars_x', 'review_count', 'is_open', 'attributes', 'categories', 'hours','composite_score']].drop_duplicates(), on='business_id', how='left')
        top_10_coffee_shops = top_10_coffee_shops.drop([2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
        # rename composite score column and drop last column
        top_10_coffee_shops = top_10_coffee_shops.rename(columns={"composite_score_x": "composite_score"})
        top_10_coffee_shops = top_10_coffee_shops.iloc[:, :-1]
        top_10_coffee_shops.reset_index(drop=True)
        return top_10_coffee_shops

if __name__ == "__main__":
    absaF = absaFilter()
    absaF.loadReviews()
    absaF.calPSS()
    top10 = absaF.calcTop10()

    absaF.reviews.to_csv('philly_reviews_asba.csv')
    absaF.review_rates.to_csv('philly_reviews_rates.csv')
    top10.to_csv('top_10_coffee_shops.csv')




