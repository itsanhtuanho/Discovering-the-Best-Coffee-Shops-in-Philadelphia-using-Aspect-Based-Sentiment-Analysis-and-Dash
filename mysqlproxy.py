import pymysql
from sqlalchemy import create_engine
import pandas as pd

class MySQLProxy:
    def __init__(self, dbname = 'davdbcoffeerebrewer'):
        """ Init the class with the dbname, all other information are now still hard-coded.
        The DB will be created automatically if it doesn't exist
        """
        self.dbname = dbname
        conn1 = pymysql.connect(host="davdbcoffeerebrewer.c3ruoumhmoer.us-west-1.rds.amazonaws.com", port=3306,
                                    user="admin", password="davdbcoffeerebrewer", db="")
        stdb1 = conn1.cursor()
        stdb1.execute("CREATE DATABASE IF NOT EXISTS " + self.dbname)
        self.conn = pymysql.connect(host="davdbcoffeerebrewer.c3ruoumhmoer.us-west-1.rds.amazonaws.com", port=3306,
                                    user="admin", password="davdbcoffeerebrewer", db=dbname)
        self.stdb = self.conn.cursor()
        
        self.sqlEngine = create_engine('mysql+pymysql://admin:davdbcoffeerebrewer' + 
                                       '@davdbcoffeerebrewer.c3ruoumhmoer.us-west-1.rds.amazonaws.com/'+dbname, pool_recycle=3600)
        self.dbConnection = self.sqlEngine.connect()

    def create_database(self):
        """
        Create the database table if it's not exists.
        """
        str1 = """CREATE TABLE IF NOT EXISTS coffeeshops ( 
        shopid varchar(32),
        opendate DATE,
        rate DOUBLE
        )
        """
        self.stdb.execute(str1)

    def create_filter_table(self):
        """
        import the filtered result into the table.
        """
        str0 = """
            CREATE TABLE IF NOT EXISTS business_info (
            id integer,
            business_id varchar(64),
            name varchar(128),
            address varchar(128),
            city varchar(64),
            state varchar(32),
            postal_code varchar(32),
            latitude double,
            longitude double,
            stars double,
            review_count integer,
            is_open boolean,
            attributes text,
            categories varchar(32),
            hours text,
            INDEX(name)
            )
        """
        self.stdb.execute(str0)
        str1 = """CREATE TABLE IF NOT EXISTS filtered_philly_reviews (
        id integer,
        business_id varchar(64),
        name varchar(128),
        address varchar(128),
        city varchar(64),
        state varchar(32),
        postal_code varchar(32),
        latitude double,
        longitude double,
        stars_x double,
        review_count integer,
        is_open boolean,
        attributes text,
        categories varchar(32),
        hours text,
        review_id varchar(64),
        user_id varchar(64),
        stars_y double,
        useful integer,
        funny integer,
        cool integer,
        review_text text,
        review_date date,
        INDEX(name)
        )
        """
        self.stdb.execute(str1)

        str1 = """CREATE TABLE IF NOT EXISTS philly_reviews_asba (
            id integer,
            business_id varchar(64),
            name varchar(128),
            address varchar(128),
            city varchar(64),
            state varchar(32),
            postal_code varchar(32),
            latitude double,
            longitude double,
            stars_x double,
            review_count integer,
            is_open boolean,
            attributes text,
            categories varchar(32),
            hours text,
            review_id varchar(64),
            user_id varchar(64),
            stars_y double,
            useful integer,
            funny integer,
            cool integer,
            review_text text,
            review_date date,
            pss double,
            positive integer,
            metric_1 double,
            metric_2 double,
            composite_score double,
            INDEX(name)

        )
        """
        self.stdb.execute(str1)

        str1 = """CREATE TABLE IF NOT EXISTS philly_shop_asba (
            id integer,
            business_id varchar(64),
            metric_1 double,
            metric_2 double,
            stars double,
            recent_stars double,
            INDEX(business_id)
            )
        """
        self.stdb.execute(str1)

    def import_philly_reviews(self):
        """
        import the philly reviews into the db
        """
        df = pd.read_csv('./Data/filtered_philly_reviews.csv')
        df.to_sql("filtered_philly_reviews", self.dbConnection, if_exists="append", index=False)

        df2 = pd.read_csv('./Data/philly_reviews_asba.csv')
        df2.to_sql('philly_reviews_asba', self.dbConnection, if_exists='append', index=False)

        df3 = pd.read_csv('./Data/business_info.csv')
        df3.rename(columns={"Unnamed: 0":"id"}, inplace=True)
        print(df3.head(5))
        df3.to_sql('business_info', self.dbConnection, if_exists='append', index=False)

        df4 = pd.read_csv('./Data/philly_reviews_rates.csv')
        df4.rename(columns={"Unnamed: 0":"id"}, inplace=True)
        df4.to_sql('philly_shop_asba', self.dbConnection, if_exists='append', index=False )

        self.dbConnection.close()

    
    def read_data(self, tb='filtered_philly_reviews', whereStr = ''):
        """
        Read data as pandas from database.
        """
        resdf = pd.read_sql("select * from " + tb +" " +  whereStr, self.dbConnection);
        return resdf

    def show_database(self):
        """
        show all the tables in the DB
        """
        print("SHOW TABLES IN DB")
        self.stdb.execute("SHOW TABLES")
        tables = self.stdb.fetchall()
        print(tables)

    def clean_database(self):
        """
        clean all the tables
        """
        self.stdb.execute("DELETE FROM coffeeshops")
        self.stdb.execute("DELETE FROM filtered_philly_reviews")
        self.stdb.execute("DELETE FROM philly_reviews_asba")
        self.stdb.execute('DELETE FROM philly_shop_asba')
        self.stdb.execute('DELETE FROM business_info')
    def add_shop_dataframe(self, df):
        """
        exmaple codes to add data frame into db
        """
        df.to_sql("coffeeshops", self.dbConnection, if_exists="append", index=False)

    def add_shop_records(self, shoplist):
        """
        add stock information, the input is a list of dict, the dict is column:value pair
        """
        str1 = "INSERT INTO coffeeshops(shopid, opendate, rate) VALUES "
        start = True
        for item in shoplist:
            if not start:
                str1 = str1 + ","
            str1 = str1 + "("
            str1 = str1 + "\'" + item["shopid"] + "\',"
            str1 = str1 + "\'" + item["opendate"] + "\',"
            str1 = str1 + str(item["rate"])
            str1 = str1 + ")"
            start = False
        
        print(str1)
        self.stdb.execute(str1)
    
    def print_all_shops(self):
        """
        Testing only, print all the records
        """
        str1 = "SELECT * FROM coffeeshops"
        self.stdb.execute(str1)
        all_shops = self.stdb.fetchall()
        print("SHOW all Shops")
        print(all_shops)


if __name__ == "__main__":
    proxy = MySQLProxy()
    proxy.show_database()
    proxy.create_database()
    proxy.show_database()
    #proxy.clean_database()
    proxy.create_filter_table()
    proxy.import_philly_reviews()
    #df1=proxy.read_data(whereStr=" where name = \"Vineyards Cafe\"")
    #print(df1)
