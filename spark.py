from datetime import date
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import (StructType,
                               StructField,
                               DoubleType,
                               IntegerType,
                               StringType,
                               BooleanType)


holidays = [date(2016, 1, 1), date(2016, 1, 18), date(2016, 3, 27), date(2016, 5, 30),
            date(2016, 7, 4),  date(2016, 9, 5), date(2016, 10, 10), date(2016, 11, 11),
            date(2016, 11, 24), date(2016, 12, 25)]

def nearest_holiday(year, month, day):
    d = date(year, month, day)
    dists = [abs(d- holiday).days for holiday in holidays]
    return min(dists)

# add a boolean column that indicates whether flight delayed or not (threshold 15 mins)
was_delayed_udf = udf(lambda x: x >= 15, BooleanType())

# convert hours, e.g. 1430 --> 14
get_hour_udf = udf(lambda x: x // 100, IntegerType())

# add column that indicates how close a flight is to a holiday
nearest_holiday_udf = udf(nearest_holiday, IntegerType())




if __name__ == "__main__":

    spark = SparkSession.builder \
        .master('local') \
        .appName('Flight Delay') \
        .getOrCreate()


    flight_data = spark.read \
        .format('com.databricks.spark.csv') \
        .csv('/home/william/Projects/flight-delay/data/merged/2016.csv',
             inferSchema='true', nanValue="", header='true', mode='PERMISSIVE')

    flight_data = flight_data \
        .withColumn('Year', flight_data['Year'].cast('int')) \
        .withColumn('Month', flight_data['Month'].cast('int')) \
        .withColumn('Day', flight_data['Day'].cast('int')) \
        .withColumn('CRSDepTime', flight_data['CRSDepTime'].cast('int')) \
        .withColumn('Dow', flight_data['Dow'].cast('int')) \
        .withColumn('DepTime', flight_data['DepTime'].cast('int')) \
        .withColumn('DepDelay', flight_data['DepDelay'].cast('int')) \
        .withColumn('TaxiOut', flight_data['TaxiOut'].cast('int')) \
        .withColumn('TaxiIn', flight_data['TaxiIn'].cast('int')) \
        .withColumn('CRSArrTime', flight_data['CRSArrTime'].cast('int')) \
        .withColumn('ArrTime', flight_data['ArrTime'].cast('int')) \
        .withColumn('ArrDelay', flight_data['ArrDelay'].cast('int')) \
        .withColumn('Cancelled', flight_data['Cancelled'].cast('int')) \
        .withColumn('Diverted', flight_data['Diverted'].cast('int')) \
        .withColumn('CRSElapsedTime', flight_data['CRSElapsedTime'].cast('int')) \
        .withColumn('ActualElapsedTime', flight_data['ActualElapsedTime'].cast('int')) \
        .withColumn('AirTime', flight_data['AirTime'].cast('int')) \
        .withColumn('Distance', flight_data['Distance'].cast('int')) \
        .withColumn('CarrierDelay', flight_data['CarrierDelay'].cast('int')) \
        .withColumn('WeatherDelay', flight_data['WeatherDelay'].cast('int')) \
        .withColumn('NASDelay', flight_data['NASDelay'].cast('int')) \
        .withColumn('SecurityDelay', flight_data['SecurityDelay'].cast('int')) \
        .withColumn('LateAircraftDelay ', flight_data['LateAircraftDelay '].cast('int'))


    flight_data = flight_data \
        .dropna(subset=['DepDelay']) \
        .filter(flight_data['Cancelled'] == 0)

    # add new udf computed columns
    flight_data = flight_data \
        .withColumn('DepDelayed', was_delayed_udf(flight_data['DepDelay'])) \
        .withColumn('CRSDepTime', get_hour_udf(flight_data['CRSDepTime'])) \
        .withColumn('NearestHoliday', nearest_holiday_udf(flight_data['Year'],
                                                          flight_data['Month'],
                                                          flight_data['Day']))

    # columns used in the predictive models
    cols = ['DepDelay', 'Month', 'Day', 'Dow', 'CRSDepTime', 'Distance', 'Carrier',
            'Origin', 'Dest', 'NearestHoliday', 'DepDelayed']

    # rename columns
    flights = flight_data \
        .select(*cols) \
        .withColumnRenamed('DepDelay', 'Delay') \
        .withColumnRenamed('CRSDepTime', 'hour') \
        .withColumnRenamed('DepDelayed', 'Delayed')

    flights.show(25)





