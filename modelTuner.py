# Tune the model weight
import pandas as pd
import mysqlproxy
from sklearn.linear_model import LinearRegression

class modelTuner(object):
    def __init__(self, filepath, isSample):
        self.shop_absa = None
        self.test_shops = None
        self.sample_reviews = None
        if not isSample:
            self.shop_absa = pd.read_csv(filepath)
            self.generateSamples(20)
        else:
            self.test_shops = pd.read_csv(filepath)

        self.getSampleReviewData()

    def generateSamples(self, sz):
        shop_with_less_reviews = self.shop_absa[self.shop_absa['review_count'] < 50]
        self.test_shops = shop_with_less_reviews.sample(sz)
        self.test_shops.to_csv('shop_samples.csv')


    def getSampleReviewData(self):
        bids = self.test_shops["business_id"].values.tolist()
        wherestr = " where business_id in " + str(tuple(bids))
        proxy = mysqlproxy.MySQLProxy()
        self.sample_reviews = proxy.read_data('philly_reviews_asba', whereStr=wherestr)
        # print(df1)
        self.sample_reviews = self.sample_reviews[["business_id", "name", "review_text", "useful","pss", "positive", "metric_1", "metric_2", "stars_x", "stars_y"]]
        self.calcWeightedASBA()
        self.sample_reviews.to_csv('shop_review_sample.csv')

    def calcWeightedASBA(self):
        self.sample_reviews["updated_useful"] = 1 + self.sample_reviews["useful"]
        self.sample_reviews['weighted_pss'] = self.sample_reviews["pss"] * self.sample_reviews['updated_useful']
        self.sample_reviews['weighted_positive'] = self.sample_reviews["positive"] * self.sample_reviews['updated_useful']

        sum_pss_byid = self.sample_reviews.groupby('business_id')['weighted_pss'].sum().reset_index()
        sum_pss_byid.columns = ['business_id', 'sum_pss']
        # Group by business and calculate the positive sentiment ratio (Metric 2)
        sum_positive_byid = self.sample_reviews.groupby('business_id')['weighted_positive'].sum().reset_index()
        sum_positive_byid.columns = ['business_id', 'sum_positive']

        sum_useful_byid = self.sample_reviews.groupby('business_id')['updated_useful'].sum().reset_index()
        sum_useful_byid.columns = ['business_id', 'sum_useful']

        res = pd.merge(sum_pss_byid, sum_positive_byid, on='business_id')
        res = pd.merge(res, sum_useful_byid, on='business_id')
        res['weighted_metric1'] = res['sum_pss'] / res['sum_useful']
        res['weighted_metric2'] = res['sum_positive'] / res['sum_useful']
        self.test_shops = pd.merge(self.test_shops, res, on='business_id')
        print(res.head())
        print(self.test_shops.head())



    def calcBestWeight(self):
        x_train = self.test_shops[["metric_1", "metric_2"]]
        y_train = self.test_shops["stars"]
        model = LinearRegression().fit(x_train, y_train)
        y_predict = model.predict(x_train)
        print(y_predict)
        return y_predict

    def compareWithDefaultFormula(self):
        # data1 = self.test_shops[["metric_1", "metric_2", "stars", "recent_stars", "human_rank"]]
        self.test_shops["calc_score"] = 3 * self.test_shops["metric_1"] + 2 * self.test_shops["metric_2"]
        self.test_shops['calc_score_weighted'] = 3 * self.test_shops['weighted_metric1'] + 2 * self.test_shops["weighted_metric2"]
        self.test_shops["predict_score"] = self.calcBestWeight()
        self.test_shops["stars_rank"] = self.test_shops["stars"].rank()
        self.test_shops["score_rank"] = self.test_shops["calc_score"].rank()
        self.test_shops["recent_star_rank"] = self.test_shops["recent_stars"].rank()
        self.test_shops["predict_rank"] = self.test_shops["predict_score"].rank()
        self.test_shops['weighted_score_rank'] = self.test_shops['calc_score_weighted'].rank()
        # print("Rank Calc:", data1["score_rank"])
        # print(data1.head())
        c1, c2, c3, c4, c5, c6, c7 = self.calcRankCorrelation(self.test_shops)
        print("Correlation between stars rank and calc score: ", c1)
        print("Correlation between recent stars rank and calc score: ", c2)
        print("Correlation between stars rank and predict score: ", c3)
        print("Correlation between recent stars rank and predict score: ", c4)
        print("Correlation between human rank and calc score: ", c5)
        print("Correlation between human rank and predict score: ", c6)
        print("Correlation between human rank and weighted calc score: ", c7)

    def calcRankCorrelation(self, dfranks):
        """ Spearman’s Rank Correlation. ranks the dataframe with ranks"""
        sumdiff_stars_score = 0
        sumdiff_rstars_score = 0
        sumdiff_rstars_predict = 0
        sumdiff_stars_predict = 0
        sumdiff_human_score = 0
        sumdiff_human_predict = 0
        sumdiff_human_weighted = 0
        n = dfranks.shape[0]
        for index, row in dfranks.iterrows():
            d1 = row['stars_rank'] - row['score_rank']
            d2 = row['recent_star_rank'] - row['score_rank']
            d3 = row['predict_rank'] - row['stars_rank']
            d4 = row['predict_rank'] - row['recent_star_rank']
            d5 = row['human_rank'] - row['score_rank']
            d6 = row['human_rank'] - row['predict_rank']
            d7 = row['human_rank'] - row['weighted_score_rank']
            dsq1 = d1 * d1
            dsq2 = d2 * d2
            dsq3 = d3 * d3
            dsq4 = d4 * d4
            dsq5 = d5 * d5
            dsq6 = d6 * d6
            dsq7 = d7 * d7
            sumdiff_stars_score = sumdiff_stars_score + dsq1
            sumdiff_rstars_score = sumdiff_rstars_score + dsq2
            sumdiff_stars_predict = sumdiff_stars_predict + dsq3
            sumdiff_rstars_predict = sumdiff_rstars_predict + dsq4
            sumdiff_human_score = sumdiff_human_score + dsq5
            sumdiff_human_predict = sumdiff_human_predict + dsq6
            sumdiff_human_weighted = sumdiff_human_weighted + dsq7

        res1 = 1 - 6 * sumdiff_stars_score /(n*(n*n -1))
        res2 = 1 - 6 * sumdiff_rstars_score / (n *(n*n -1))
        res3 = 1- 6 * sumdiff_stars_predict / (n * (n*n -1))
        res4 = 1 - 6 * sumdiff_rstars_predict / (n * (n * n -1))
        res5 = 1 - 6 * sumdiff_human_score / (n * (n * n -1))
        res6 = 1 - 6 * sumdiff_human_predict / (n * (n * n -1))
        res7 = 1 - 6 * sumdiff_human_weighted / (n * (n * n -1))
        return res1, res2, res3, res4, res5, res6, res7

if __name__ == "__main__":
    # mt = modelTuner("./Data/philly_reviews_rates.csv", False)
    # mt.generateSamples(30)
    mt = modelTuner("./Data/shop_samples.csv", True)
    mt.calcBestWeight()
    mt.compareWithDefaultFormula()