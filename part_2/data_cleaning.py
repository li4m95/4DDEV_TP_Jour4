import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

directoryRaw = "DataSet_csv"
cleaned_directory = "DataSet_cleaned_csv"

spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

def read_data_from_csv(file_path):
    return spark.read.option("header", True).csv(file_path)

def get_list_of_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

FICTIONAL_COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Umbrella Corp", "Hooli",
    "Dunder Mifflin", "Stark Industries", "Wayne Enterprises"
]

def clean_data(df):
    df = df.dropDuplicates()

    if "Company" in df.columns:
        companies_array = F.array([F.lit(c) for c in FICTIONAL_COMPANIES])
        random_company = companies_array[F.floor(F.rand() * len(FICTIONAL_COMPANIES)).cast("int")]
        df = df.withColumn("Company",
            F.when(F.col("Company").isNull(), random_company)
             .otherwise(F.col("Company"))
        )

    fill_cols = [c for c in ["Email", "Country"] if c in df.columns]
    if fill_cols:
        df = df.fillna("Unknown", subset=fill_cols)

    for col in ["FirstName", "LastName"]:
        if col in df.columns:
            df = df.withColumn(col, F.initcap(F.col(col)))

    if "LastName" in df.columns:
        df = df.orderBy("LastName")

    return df


if __name__ == "__main__":
    csv_files = get_list_of_csv_files(directoryRaw)
    print("List of CSV files in the directory:")
    print(csv_files)
    for csv_file in csv_files:
        file_path = os.path.join(directoryRaw, csv_file)
        df = read_data_from_csv(file_path)
        df = clean_data(df)
        print(f"Data from {csv_file}:")
        df.show(5)

    spark.stop()
