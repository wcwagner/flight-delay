import os
from datetime import date
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression


if __name__ == '__main__':
    spark = SparkSession.builder \
        .master('local') \
        .appName('Flight Delay') \
        .getOrCreate()
    # read in the pre-processed DataFrame from the parquet file
    base_dir = '/home/william/Projects/flight-delay/data/parquet'
    flights_df = spark.read.parquet(os.path.join(base_dir, 'flights.parquet'))

    # categorical columns that will be OneHotEncoded
    cat_cols = ['Month', 'Day', 'Dow', 'Hour', 'Carrier']
    # numeric columns that will be a part of features used for prediction
    non_cat_cols = ['Delay', 'Distance', 'NearestHoliday']
    # StringIndexer does not have multiple col support yet (PR #9183 )
    # Create StringIndexer for each categorical feature
    indexers = [ StringIndexer(inputCol=col, outputCol=col+'_Index')
                 for col in cat_cols ]
    # OneHotEncode each categorical feature after being StringIndexed
    encoders = [ OneHotEncoder(dropLast=False, inputCol=indexer.getOutputCol(),
                               outputCol=indexer.getOutputCol()+'_Encoded')
                 for indexer in indexers ]
    # Assemble all feature columns (numeric + categorical) into `features` col
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol()
                                           for encoder in encoders] + non_cat_cols,
                                outputCol='features')
    # Create pipeline to process all StringIndex, OneHotEncoder, and assmelber operations
    pipeline = Pipeline(stages=[ *indexers, *encoders, assembler ] )
    flights_df = pipeline.fit(flights_df) \
                    .transform(flights_df)

    flights_df \
        .select( *(cat_cols + non_cat_cols + ['features']) ) \
        .show(10)
