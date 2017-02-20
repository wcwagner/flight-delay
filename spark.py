from datetime import date
from pyspark.sql import SQLContext, SparkSession


holidays = [date(2016, 1, 1), date(2016, 1, 18), date(2016, 3, 27), date(2016, 5, 30),
            date(2016, 7, 4),  date(2016, 9, 5), date(2016, 10, 10), date(2016, 11, 11),
            date(2016, 11, 24), date(2016, 12, 25)]


if __name__ == "__main__":

    sess = SparkSession.builder \
        .master('local') \
        .appName('Flight Delay') \
        .getOrCreate()


    flights = sess.read \
        .format('com.databricks.spark.csv') \
        .csv('/home/william/Projects/flight-delay/data/merged/2016.csv',
             inferSchema='true', nanValue="", header='true', mode='PERMISSIVE')

    # Using schema didn't work. So let `inferSchema` set all columns to strings
    # then manually cast the numeric columns
    flights = flights \
        .withColumn('Year', flts['Year'].cast('int')) \
        .withColumn('Month', flts['Month'].cast('int')) \
        .withColumn('Day', flts['Day'].cast('int')) \
        .withColumn('Dow', flts['Dow'].cast('int')) \
        .withColumn('CRSDepTime', flts['CRSDepTime'].cast('int')) \
        .withColumn('DepTime', flts['DepTime'].cast('int')) \
        .withColumn('DepDelay', flts['DepDelay'].cast('int')) \
        .withColumn('TaxiOut', flts['TaxiOut'].cast('int')) \
        .withColumn('TaxiIn', flts['TaxiIn'].cast('int')) \
        .withColumn('CRSArrTime', flts['CRSArrTime'].cast('int')) \
        .withColumn('ArrTime', flts['ArrTime'].cast('int')) \
        .withColumn('ArrDelay', flts['ArrDelay'].cast('int')) \
        .withColumn('Cancelled', flts['Cancelled'].cast('int')) \
        .withColumn('Diverted', flts['Diverted'].cast('int')) \
        .withColumn('CRSElapsedTime', flts['CRSElapsedTime'].cast('int')) \
        .withColumn('ActualElapsedTime', flts['ActualElapsedTime'].cast('int')) \
        .withColumn('AirTime', flts['AirTime'].cast('int')) \
        .withColumn('Distance', flts['Distance'].cast('int')) \
        .withColumn('CarrierDelay', flts['CarrierDelay'].cast('int')) \
        .withColumn('WeatherDelay', flts['WeatherDelay'].cast('int')) \
        .withColumn('NASDelay', flts['NASDelay'].cast('int')) \
        .withColumn('SecurityDelay', flts['SecurityDelay'].cast('int')) \
        .withColumn('LateAircraftDelay ', flts['LateAircraftDelay '].cast('int'))




