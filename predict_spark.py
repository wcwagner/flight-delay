import os
from datetime import date
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression

if __name__ == '__main__':
    spark = SparkSession.builder \
        .master('local') \
        .appName('Flight Delay') \
        .getOrCreate()
    # read in the pre-processed DataFrame from the parquet file
    base_dir = '/home/william/Projects/flight-delay/data/parquet'
    flights_df = spark.read.parquet(os.path.join(base_dir, 'flights.parquet'))

    # encode the categorical variables
    str_cat_cols = ['Carrier', 'Origin', 'Dest']
    # StringIndexer does not have multiple col support yet (PR #9183 )
    indexers = [ StringIndexer(inputCol=col, outputCol=col+'_index').fit(flights_df)
                 for col in str_cat_cols ]
    str_ixer_pipeline = Pipeline(stages=indexers)
    flights_df = str_ixer_pipeline.fit(flights_df).transform(flights_df)

    flights_df.show(10)

    numeric_cats = ['Month', 'Day', 'Dow', 'Hour', 'HDays']

    #lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    #lr_model = lr.fit(flights_df)
    print("Total rows: ", flights_df.count())
