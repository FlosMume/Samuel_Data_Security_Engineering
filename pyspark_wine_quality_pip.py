# ===================================================================================
# FILE: pyspark_wine_quality_pipeline.py
# DESCRIPTION: End-to-end ML pipeline using PySpark
# TOOLS: Spark MLlib, PySpark SQL, Transformers, Estimators
# DATASET: Wine Quality (Red Wine)
# STEPS: Load → Clean → Transform → Train → Evaluate → Predict
# ===================================================================================

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Step 1: Initialize Spark Session
# --------------------------------
# SparkSession is the entry point to Spark functionality.
# It allows you to create DataFrames, run SQL queries, and use MLlib.
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Optional: Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Step 2: Load Data from CSV
# ---------------------------
# We load the red wine dataset. The data is semi-colon separated.
# inferSchema=True tells Spark to automatically detect column types.
# header=True uses the first row as column names.
df = spark.read.option("sep", ";").csv(
    "winequality-red.csv", 
    header=True, 
    inferSchema=True
)

# Display schema and first few rows
df.printSchema()
df.show(5)

# Step 3: Clean and Prepare Data using Spark SQL
# ------------------------------------------------
# Register the DataFrame as a temporary SQL view for easy querying.
df.createOrReplaceTempView("wine_data")

# Use Spark SQL to inspect data and handle potential issues
# For example, filter out rows where quality is null or invalid
clean_df = spark.sql("""
    SELECT *
    FROM wine_data
    WHERE quality IS NOT NULL
      AND alcohol > 0 AND pH > 0
""")

print("After filtering invalid rows:")
clean_df.show(5)

# Step 4: Feature Engineering
# ----------------------------
# We'll use VectorAssembler to combine input features into a single vector column.
# This is required by most ML algorithms in Spark MLlib.

# List of input features (all columns except 'quality')
feature_columns = [col for col in clean_df.columns if col != "quality"]

# Assemble features into a single vector column called "features"
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features",  # raw un-scaled features
    handleInvalid="skip"       # skip rows with missing values
)

# Step 5: Scale Features
# -----------------------
# Many ML algorithms perform better with normalized features.
# StandardScaler standardizes features to have zero mean and unit variance.
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Step 6: Define the ML Model
# ----------------------------
# We'll use Linear Regression to predict wine quality (a continuous variable).
lr = LinearRegression(
    featuresCol="features",
    labelCol="quality",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.01
)

# Step 7: Build the ML Pipeline
# ------------------------------
# A Pipeline chains multiple stages (transformers and estimators).
# This ensures that the same preprocessing is applied during training and inference.
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Step 8: Split Data into Training and Test Sets
# -----------------------------------------------
train_data, test_data = clean_df.randomSplit([0.8, 0.2], seed=1234)

# Step 9: Hyperparameter Tuning with Cross-Validation
# ----------------------------------------------------
# We'll tune the regularization parameter (regParam) of Linear Regression.
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Define evaluation metric
evaluator = RegressionEvaluator(
    labelCol="quality",
    predictionCol="prediction",
    metricName="rmse"
)

# Cross-validator: 3-fold cross-validation
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2  # Run 2 parameter combinations at once
)

# Step 10: Train the Model
# -------------------------
# Fit the cross-validator on training data
model = cv.fit(train_data)

# Step 11: Make Predictions
# --------------------------
predictions = model.transform(test_data)

# Show predictions vs actual
predictions.select("quality", "prediction").show(10)

# Step 12: Evaluate the Model
# ----------------------------
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Optional: Get R² (coefficient of determination)
r2_evaluator = RegressionEvaluator(
    labelCol="quality",
    predictionCol="prediction",
    metricName="r2"
)
r2 = r2_evaluator.evaluate(predictions)
print(f"R² Score: {r2:.4f}")

# Step 13: Save the Model (Optional)
# -----------------------------------
# model.bestModel.write().overwrite().save("wine_quality_model")

# Step 14: Stop Spark Session
# ----------------------------
spark.stop()

# ✅ Summary:
# - We used PySpark to build a full ML pipeline.
# - Spark SQL helped clean and inspect data.
# - VectorAssembler and StandardScaler prepared features.
# - LinearRegression trained the model.
# - CrossValidator tuned hyperparameters.
# - The pipeline ensures reproducibility.

# 🔗 References:
# 1. Spark MLlib Guide: https://spark.apache.org/docs/latest/ml-guide.html
# 2. PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
# 3. ML Pipelines: https://spark.apache.org/docs/latest/ml-pipeline.html
# 4. Wine Quality Dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality