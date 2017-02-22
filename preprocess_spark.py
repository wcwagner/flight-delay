import os
from datetime import timedelta, date
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import (StructType,
                               StructField,
                               DoubleType,
                               IntegerType,
                               StringType,
                               BooleanType)
# Selected holidays for the 2015 & 2016 calendar year -- to be used for Nearest Holiday
holidays_2015 = [
    date(2015, 1, 1), date(2015, 1, 19), date(2015, 4, 5), date(2015, 5, 25),
    date(2015, 7, 4), date(2015, 9, 7), date(2015, 10, 12), date(2015, 11, 11),
    date(2015, 11, 26), date(2015, 12, 25)
]
holidays_2016 = [
    date(2016, 1, 1), date(2016, 1, 18), date(2016, 3, 27), date(2016, 5, 30),
    date(2016, 7, 4),  date(2016, 9, 5), date(2016, 10, 10), date(2016, 11, 11),
    date(2016, 11, 24), date(2016, 12, 25)
]
# utility function to loop thru each day in year
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
# lookup table that matches each date, for the years speicfied above, to the
# the number of days until the nearest holiday

hday_lookup = {}
for d in daterange( date(2015, 1, 1), date(2017, 1, 1) ):
    holidays = holidays_2015 if d.year == 2015 else holidays_2016
    hday_lookup[d] = min( (float(abs(d- holiday).days) for holiday in holidays) )

def nearest_holiday(year, month, day):
    d = date(int(year), int(month), int(day))
    return hday_lookup[d]

# add a boolean column that indicates whether flight delayed or not (threshold 15 mins)
was_delayed_udf = udf(lambda x: float(x >= 15), DoubleType())

# convert hours, e.g. 1430 --> 14
get_hour_udf = udf(lambda x: float(x // 100), DoubleType())

# add column that indicates how close a flight is to a holiday
nearest_holiday_udf = udf(nearest_holiday, DoubleType())


if __name__ == "__main__":

    spark = SparkSession.builder \
        .master('local') \
        .appName('Flight Delay') \
        .getOrCreate()


    flight_data = spark.read \
        .format('com.databricks.spark.csv') \
        .csv('/home/william/Projects/flight-delay/data/merged/allyears.csv',
             inferSchema='true', nanValue="", header='true', mode='PERMISSIVE')
    # there is a PR to accept multiple `nanValue`s, until then, however, the schema
    # must be manually cast (due to the way the DOT stores the data)
    flight_data = flight_data \
        .withColumn('Year', flight_data['Year'].cast('int')) \
        .withColumn('Month', flight_data['Month'].cast('Double')) \
        .withColumn('Day', flight_data['Day'].cast('Double')) \
        .withColumn('CRSDepTime', flight_data['CRSDepTime'].cast('Double')) \
        .withColumn('Dow', flight_data['Dow'].cast('Double')) \
        .withColumn('DepTime', flight_data['DepTime'].cast('Double')) \
        .withColumn('DepDelay', flight_data['DepDelay'].cast('Double')) \
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
        .withColumn('Distance', flight_data['Distance'].cast('Double')) \
        .withColumn('CarrierDelay', flight_data['CarrierDelay'].cast('int')) \
        .withColumn('WeatherDelay', flight_data['WeatherDelay'].cast('int')) \
        .withColumn('NASDelay', flight_data['NASDelay'].cast('int')) \
        .withColumn('SecurityDelay', flight_data['SecurityDelay'].cast('int')) \
        .withColumn('LateAircraftDelay', flight_data['LateAircraftDelay'].cast('int'))

    # drop cancelled flights, and flights where there is no departure delay data
    flight_data = flight_data \
        .dropna(subset=['DepDelay']) \
        .filter(flight_data['Cancelled'] == 0)

    # add new udf computed columns
    flight_data = flight_data \
        .withColumn('Delayed', was_delayed_udf(flight_data['DepDelay'])) \
        .withColumn('CRSDepTime', get_hour_udf(flight_data['CRSDepTime'])) \
        .withColumn('HDays', nearest_holiday_udf(flight_data['Year'],
                                                 flight_data['Month'],
                                                 flight_data['Day']))

    # columns used in the predictive models
    cols = ['DepDelay', 'Month', 'Day', 'Dow', 'CRSDepTime', 'Distance', 'Carrier',
            'Origin', 'Dest', 'HDays', 'Delayed']

    # rename columns
    flights = flight_data \
        .select(*cols) \
        .withColumnRenamed('DepDelay', 'Delay') \
        .withColumnRenamed('CRSDepTime', 'Hour')

    print("Table before storing")
    flights.show(5)

    base_data_path = '/home/william/Projects/flight-delay/data'
    if not os.path.exists(os.path.join(base_data_path, 'parquet')):
        os.makedirs(os.path.join(base_data_path, 'parquet'))
    flights.write.parquet(os.path.join(base_data_path, 'parquet', 'flights.parquet'),
                                       mode='overwrite')




