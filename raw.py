from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from utilities.utils import *


# Entry point of the program
if __name__ == "__main__":
    """
    Main function to run the program.
    """

    # Create a SparkSession
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    clickStreamSchema = StructType([
        StructField("ClickID", StringType(), True),
        StructField("UserID", StringType(), True),
        StructField("SessionID", StringType(), True),
        StructField("Timestamp", TimestampType(), True),
        StructField("PageURL", StringType(), True),
        StructField("Action", StringType(), True),
        StructField("ProductID", StringType(), True),
        StructField("Category", StringType(), True),
        StructField("Referrer", StringType(), True),
        StructField("Device", StringType(), True),
        StructField("Location", StringType(), True)
    ])

    # Read the JSON file containing clickstream data
    clickStream_df = spark.read.json(
        r"F:\Project E-Commerce\Website and User Experience (UX) Analytics\clickstream_data_large.json", schema=clickStreamSchema)


    # Split the 'location' column into 'city' and 'country' columns
    clickStream_df = CommonUtils().split_location(clickStream_df)

    # Check for null values in the DataFrame and replace them with 'NA'
    df = CommonUtils().NullCheck(clickStream_df)

    # Check for empty strings in the DataFrame and replace them with 'NA'
    clickStream_df = CommonUtils().empty_string_check(df)

    # Show the resulting DataFrame
    # clickStream_df.show()

    # Read the CSV file containing A/B testing data
    testing_df = spark.read.csv(
        r"F:\Project E-Commerce\Website and User Experience (UX) Analytics\ab_testing_data_large.csv", header=True, inferSchema=True)

    # Split the 'location' column into 'city' and 'country' columns
    testing_df = CommonUtils().split_location(testing_df)

    # Check for null values in the DataFrame and replace them with 'NA'
    df = CommonUtils().NullCheck(testing_df)

    # Check for empty strings in the DataFrame and replace them with 'NA'
    testing_df = CommonUtils().empty_string_check(df)

    # Show the resulting DataFrame
    # testing_df.show()

    # Join the clickStream_df and testing_df DataFrames on UserID and SessionID columns,
    # select the required columns from the joined DataFrame and store it in unified_df
    unified_df = clickStream_df.join(testing_df, on=["userID", "sessionID"], how="inner").select(
        clickStream_df.UserID,  # User ID
        clickStream_df.SessionID,  # Session ID
        clickStream_df.ClickID,  # Click ID
        clickStream_df.Action,  # Action
        clickStream_df.Category,  # Category
        clickStream_df.Device,  # Device
        clickStream_df.PageURL,  # Page URL
        clickStream_df.Timestamp,  # Timestamp
        clickStream_df.City,  # City
        clickStream_df.Country,  # Country
        testing_df.TestGroup,  # Test Group
        testing_df.PageVersion,  # Page Version
        testing_df.Conversion,  # Conversion
        testing_df.Revenue  # Revenue
    )

    # Show the resulting DataFrame with the count of rows
    # unified_df.show(unified_df.count())

    # Perform sessionization and session analysis on the data
    # and show the resulting DataFrame with additional columns
    # and write the DataFrame to a CSV file

    # Perform sessionization and session analysis on the data
    # and return the resulting DataFrame
    df = CommonUtils().sessionization_session_analysis(clickStream_df, unified_df)

    # Split the 'sessionDuration' column into three columns:
    # 'sessionDuration_In_days', 'sessionDuration_In_hhMMss', and 'sessionDuration_In_seconds'
    # and drop the original 'sessionDuration' column
    # and show the resulting DataFrame
    df = df.withColumn("sessionDuration", split(col("sessionDuration"), "'")[1]).withColumn(
        "sessionDuration_In_days", split(col("sessionDuration"), ' ')[0]).withColumn(
            "sessionDuration_In_hhMMss", split(col("sessionDuration"), ' ')[1]).drop("sessionDuration")

    # Write the resulting DataFrame to a CSV file
    df.write.csv(path=r"F:\Project E-Commerce\output_files\sessionization and session analysis",
                 header=True, mode="overwrite")

    # Perform cross-device analysis on the unified DataFrame
    # and write the resulting DataFrame to a CSV file
    # The cross-device analysis calculates the total conversion rate and total interactions per device
    # and stores the results in a DataFrame
    # The resulting DataFrame is then written to a CSV file
    # The path to the CSV file is specified in the write.csv() method
    df = CommonUtils().cross_device_analysis(unified_df)
    df.write.csv(
        path="F:\Project E-Commerce\output_files\cross-device-analysis",  # Path to the CSV file
        header=True,  # Write the header row in the CSV file
        mode="overwrite"  # Overwrite the file if it already exists
    )

    # Perform user segmentation on the clickstream and unified data
    # The user segmentation calculates frequent users, high value users, and users with only one session with bonus revenue
    # and stores the results in a DataFrame
    # The resulting DataFrame is then written to a CSV file
    df = CommonUtils().user_segmentation(clickStream_df, testing_df)
    df.write.csv(
        path=r"F:\Project E-Commerce\output_files\user segmentation",  # Path to the CSV file
        header=True,  # Write the header row in the CSV file
        mode="overwrite"  # Overwrite the file if it already exists
    )

    # Perform conversion attribution on the clickstream and unified data
    # The conversion attribution function returns two DataFrames:
    # - df1: Contains the last page visited by the user before the conversion
    # - df2: Contains the multi-touch attribution results
    # These DataFrames are then written to CSV files
    # The paths to the CSV files are specified in the write.csv() method
    # The resulting DataFrames are stored in df1 and df2 variables
    df1, df2 = CommonUtils().conversion_attribution(clickStream_df, testing_df)
    
    # Write df1 to a CSV file
    # The path to the CSV file is specified in the write.csv() method
    # The header row is written to the CSV file
    # The file is overwritten if it already exists
    df1.write.csv(
        path="F:\Project E-Commerce\output_files\conversion attribution\last_interaction",  # Path to the CSV file
        header=True,  # Write the header row in the CSV file
        mode="overwrite"  # Overwrite the file if it already exists
    )
    
    # Write df2 to a CSV file
    # The path to the CSV file is specified in the write.csv() method
    # The header row is written to the CSV file
    # The file is overwritten if it already exists
    df2.write.csv(
        path="F:\Project E-Commerce\output_files\conversion attribution\multi_touch_attribution",  # Path to the CSV file
        header=True,  # Write the header row in the CSV file
        mode="overwrite"  # Overwrite the file if it already exists
    )

    # Stop the SparkSession
    spark.stop()
