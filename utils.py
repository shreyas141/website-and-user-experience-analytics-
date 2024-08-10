from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window


class CommonUtils:
    def NullCheck(self, df: DataFrame) -> DataFrame:
        """
        This function takes a DataFrame and checks for null values in each column.
        If a column contains null values, it replaces them with the string 'NA'.

        Parameters:
        df (pyspark.sql.DataFrame): The DataFrame to check for null values.

        Returns:
        pyspark.sql.DataFrame: The DataFrame with null values replaced.
        """
        if df is None:
            raise ValueError("The input DataFrame is None.")

        # Get the count of null values for each column
        df_null_counts = df.select(
            [count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

        # Get the columns with non-zero null counts
        column_s = [c for c in df_null_counts.columns if df_null_counts.select(c).first()[
            0] > 0]

        if len(column_s) == 0:
            return df
        else:
            print("Found null values in the following columns:", column_s)
            # Replace null values with 'NA' in each column
            for c in column_s:
                df = df.withColumn(
                    c, when(col(c).isNull(), 'NA').otherwise(col(c)))
            return df

    def split_location(self, df: DataFrame) -> DataFrame:
        """
        This function takes a DataFrame and splits the 'location' column into two columns: 'city' and 'country'.

        Parameters:
        df (pyspark.sql.DataFrame): The DataFrame to split.

        Returns:
        pyspark.sql.DataFrame: The DataFrame with the 'city' and 'country' columns.
        """
        # Check if the input DataFrame is None
        if df is None:
            raise ValueError("The input DataFrame is None.")

        # Split the 'location' column into 'City' and 'Country' columns
        df = df.withColumn("Location", split(col('location'), ","))
        df = df.withColumn("City", col('Location')[0]).withColumn("Country",
                                                                  col('Location')[1]).drop(
            "Location")
        return df

    def empty_string_check(self, df: DataFrame) -> DataFrame:
        """
        This function takes a DataFrame and checks for empty strings in each column.
        If a column contains empty strings, it replaces them with the string 'NA'.

        Parameters:
        df (pyspark.sql.DataFrame): The DataFrame to check for empty strings.

        Returns:
        pyspark.sql.DataFrame: The DataFrame with empty strings replaced.
        """
        # Get the count of empty strings for each column
        df_empty_value_cnt = df.select(
            [count(when(trim(col(c)) == '', c)).alias(c) for c in df.columns])

        # Get the columns with non-zero empty string counts
        column_s = [c for c in df_empty_value_cnt.columns if df_empty_value_cnt.select(c).first()[
            0] > 0]

        # Replace empty strings with 'NA' in each column
        for c in column_s:
            df = df.withColumn(
                c, when(trim(col(c)) == '', 'NA').otherwise(col(c)))

        return df

    def sessionization_session_analysis(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        """
        This function performs sessionization on the given DataFrame and calculates the session duration, pageviews per session, and conversion rate per session.

        Parameters:
        df1 (pyspark.sql.DataFrame): The DataFrame containing the clickstream data.
        df2 (pyspark.sql.DataFrame): The DataFrame containing the A/B testing data or unified data.

        Returns:
        pyspark.sql.DataFrame: The DataFrame with the session duration, pageviews per session, and conversion rate per session.
        """
        # Calculate session start time, session end time, and session duration
        sessionDuration_df = df1.withColumn(
            "sessionStartTime", min(df1["Timestamp"]).over(Window.partitionBy("UserID", "SessionID")))\
            .withColumn("sessionEndTime", max(df1["Timestamp"]).over(Window.partitionBy("UserID", "SessionID")))\
            .withColumn("sessionDuration", (col("sessionEndTime") - col("sessionStartTime"))) \
            .select("UserID", "SessionID", "sessionStartTime", "sessionEndTime", "sessionDuration")

        # Calculate the number of pageviews per session
        pageviews_df = df1.groupby("UserID", "SessionID").agg(
            count('pageURL').alias('PageviewsPerSession'))

        # Calculate the conversion rate per session
        conversionRate = df2.groupby('UserID', 'SessionID').agg((sum(when(col('Conversion') == True, 1).otherwise(0)) /
                                                                 count(col('sessionID'))).alias('ConversionRatePerSession'))

        # Join the DataFrames on UserID and SessionID to get the session duration, pageviews per session, and conversion rate per session
        df = sessionDuration_df.join(pageviews_df, on=["UserID", "SessionID"], how="inner") \
            .join(conversionRate, on=["UserID", "SessionID"], how="inner")

        return df

    def cross_device_analysis(self, df: DataFrame) -> DataFrame:
        """
        This function performs cross-device analysis on the given DataFrame and returns the result as a DataFrame.

        Parameters:
        df (pyspark.sql.DataFrame): The DataFrame containing the integrated data.

        Returns:
        pyspark.sql.DataFrame: The DataFrame containing the cross-device analysis results.
        """
        # Perform cross-device analysis
        users_devices_df = df.groupBy("UserID").agg(
            collect_set("Device").alias("Devices"))
        multiple_divice_users_df = users_devices_df.filter(
            size(col("Devices")) > 1)

        multiple_divice_users_df = df.join(
            multiple_divice_users_df.select('userID'), on='userID', how='inner')

        device_conversion_df = multiple_divice_users_df.groupBy('Device').agg(
            sum(when(
                col('conversion') == True, 1).otherwise(0)
                ).alias('Total_Conversion'),
            count('conversion').alias('Total_Interactions')
        )

        df = device_conversion_df.withColumn(
            'Conversion_Rate', expr('Total_Conversion / Total_Interactions'))

        return df

    def user_segmentation(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        """
        This function performs user segmentation on the given DataFrames and returns the result as a DataFrame.

        Parameters:
        df1 (pyspark.sql.DataFrame): The DataFrame containing clickstream data.
        df2 (pyspark.sql.DataFrame): The DataFrame containing A/B testing data .

        Returns:
        pyspark.sql.DataFrame: The DataFrame containing the user segmentation results.
        """
        # Perform user segmentation by finding frequent users and high value users

        # Find frequent users with more than 4 sessions
        frequet_users_df = df1.groupBy("UserID").agg(
            countDistinct('sessionID').alias('Session_Count'))
        frequet_users_df = frequet_users_df.filter("Session_Count > 4")

        # Find high value users with average revenue greater than 200

        high_value_users_df = df2.groupBy("userID").agg(
            avg('Revenue').alias('Average_Revenue'))
        high_value_users_df = high_value_users_df.filter(
            "Average_Revenue > 200")

        # Find users who have only one session with bonus revenue
        session_view_df = df1.groupBy('userID', 'sessionID').agg(
            count('pageURL').alias('Pageviews'))
        bonuce_rate_df = session_view_df.filter("Pageviews = 1")
        bonuce_rate_user = bonuce_rate_df.groupBy('userID').count().select(
            col('userID'), col('count').alias('Bonuce_Sessions'))

        # Join the DataFrames to get the user segmentation results

        df = frequet_users_df.join(high_value_users_df, 'userID', 'outer').join(
            bonuce_rate_user, 'userID', 'outer')
        return df

    def conversion_attribution(self, df1: DataFrame, df2: DataFrame) -> DataFrame:
        """
        This function performs conversion attribution on the given DataFrames and returns the result as a DataFrame.

        Parameters:
        df1 (pyspark.sql.DataFrame): The DataFrame containing the clickstream data.
        df2 (pyspark.sql.DataFrame): The DataFrame containing the A/B testing data or unified data.

        Returns:
        pyspark.sql.DataFrame: The DataFrame containing the multi touch attribution results.
        """

        # Join df1 and df2 on UserID and SessionID to create unified DataFrame
        unified_df = df1.join(df2.select('userID', 'sessionID', 'conversion'), on=[
                              "UserID", "SessionID"], how="inner")

        # Window specification for partitioning the DataFrame by UserID and SessionID and ordering by timestamp in descending order
        window_spec = Window.partitionBy(
            'userID', 'sessionID').orderBy(col('timestamp').desc())

        # Find the last page visited by the user before the conversion and filter the DataFrame to only include rows where the user converted
        last_interaction_df = unified_df.withColumn('Last_Page_Before_Conversion', last(
            'pageURL', True).over(window_spec)).filter(col('conversion') == True)

        # Select the required columns from the last_interaction_df
        last_interaction_df = last_interaction_df.select(
            'userID', 'sessionID', 'Last_Page_Before_Conversion', 'conversion')

        # Group the unified_df by UserID and SessionID and calculate the count of PageURLs to get the touchpoint count
        touchpoint_count_df = unified_df.groupBy("UserID", "SessionID").agg(
            count("PageURL").alias("Touchpoint_Count"))

        # Join the unified_df and touchpoint_count_df on UserID and SessionID
        multi_touch_df = unified_df.join(touchpoint_count_df, on=[
                                         "UserID", "SessionID"], how="inner")

        # Calculate the touchpoint credit by dividing the conversion value by the touchpoint count
        multi_touch_df = multi_touch_df.withColumn("Conversion", when(col("conversion") == True, 1).otherwise(
            0)).withColumn("Touchpoint_Credit", expr("Conversion / Touchpoint_Count"))

        # Select the required columns from the multi_touch_df and filter the DataFrame to only include rows where the user converted
        multi_touch_attribution_df = multi_touch_df.select(
            "UserID", "SessionID", "PageURL", "Touchpoint_Credit").filter(col("Conversion") == True)

        return last_interaction_df, multi_touch_attribution_df
