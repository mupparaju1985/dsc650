from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session
spark = SparkSession.builder.appName("Insurance Charges Prediction").enableHiveSupport().getOrCreate()

# Step 2: Load the data from the Hive table 'insurance'
# We select numerical features and the label 'charges'
insurance_df = spark.sql("SELECT age, bmi, children, smoker, charges FROM insurance")

# Step 3: Handle categorical data and nulls
# 'smoker' is a string ("yes"/"no"), so we convert it to a number for the ML model
indexer = StringIndexer(inputCol="smoker", outputCol="smoker_indexed")
insurance_df = indexer.fit(insurance_df).transform(insurance_df)
insurance_df = insurance_df.na.drop() 

# Step 4: Prepare data for MLlib by assembling features into a vector
# We use age, bmi, children, and our new smoker_indexed column
assembler = VectorAssembler(
    inputCols=["age", "bmi", "children", "smoker_indexed"],
    outputCol="features",
    handleInvalid="skip"
)
assembled_df = assembler.transform(insurance_df).select("features", "charges")

# Step 5: Split the data (70% training, 30% testing)
train_data, test_data = assembled_df.randomSplit([0.7, 0.3])

# Step 6: Initialize and train a Linear Regression model predicting 'charges'
lr = LinearRegression(labelCol="charges")
lr_model = lr.fit(train_data)

# Step 7 & 8: Evaluate and Print Metrics
test_results = lr_model.evaluate(test_data)
rmse_val = str(test_results.rootMeanSquaredError)
r2_val = str(test_results.r2)

print(f"Root Mean Squared Error (RMSE): {rmse_val}")
print(f"R2 Score: {r2_val}")

# ---- Objective 7: Write metrics to HBase table 'insurance_metrics' ----
data = [
    ('run1', 'cf:rmse', rmse_val),
    ('run1', 'cf:r2', r2_val),
]

def write_to_hbase_partition(partition):
    # 'master' is the host where your Thrift server is running
    connection = happybase.Connection('master') 
    connection.open()
    table = connection.table('insurance_metrics') # Updated table name
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value})
    connection.close()

# Parallelize and save to HBase
rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

spark.stop()

