# Databricks notebook source
# MAGIC %md
# MAGIC # Apache Spark Tutorial: Getting Started with Apache Spark on Databricks
# MAGIC
# MAGIC **Goal:** Learn core Spark & Databricks concepts by doing. You'll:
# MAGIC 1) generate a small dataset, 2) load it with Spark, 3) explore with DataFrames & SQL,
# MAGIC 4) run aggregations & window functions, 5) build a simple ML model,
# MAGIC 6) save & work with Delta Lake, and 7) try a tiny streaming job.
# MAGIC
# MAGIC > **How to use:** Run the cells from top to bottom. All paths use `dbfs:/` and should work on any Databricks workspace.
# MAGIC
# MAGIC **What you'll practice**
# MAGIC - SparkSession basics
# MAGIC - DataFrame API & Spark SQL
# MAGIC - Joins, aggregations, windows
# MAGIC - Caching & query plans
# MAGIC - Spark ML (Pipeline)
# MAGIC - Delta Lake (ACID tables, MERGE, time travel)
# MAGIC - Structured Streaming
# MAGIC
# MAGIC ----
# MAGIC **Tip:** If `display()` isn't available (e.g., community editions or some runtimes), fallback to `df.show(truncate=False)`.

# COMMAND ----------

# MAGIC %python
# Spark version and session checks
spark.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Notebook Config & Widgets
# MAGIC We'll parameterize a few settings using Databricks widgets, so this notebook is reusable.

# COMMAND ----------

# MAGIC %python
dbutils.widgets.removeAll()
dbutils.widgets.text("base_path", "dbfs:/tmp/spark_tutorial", "Base Path")
dbutils.widgets.dropdown("rows", "10000", ["1000", "5000", "10000", "50000"], "Synthetic Rows")
dbutils.widgets.text("catalog", "", "Unity Catalog (optional)")
dbutils.widgets.text("schema", "spark_tutorial", "Schema/Database")
dbutils.widgets.text("table", "sales_delta", "Delta Table Name")

base_path = dbutils.widgets.get("base_path")
rows      = int(dbutils.widgets.get("rows"))
catalog   = dbutils.widgets.get("catalog").strip()
schema    = dbutils.widgets.get("schema").strip()
table     = dbutils.widgets.get("table").strip()

db = f"`{schema}`" if not catalog else f"`{catalog}`.`{schema}`"
full_table_name = f"{db}.`{table}`"
silver_path = f"{base_path}/silver"
gold_path   = f"{base_path}/gold"
raw_path    = f"{base_path}/raw"

print("Config ->")
print("  base_path:", base_path)
print("  rows     :", rows)
print("  database :", db)
print("  table    :", full_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Generate a Synthetic Dataset
# MAGIC We'll create a simple retail-style dataset: orders, items, categories, prices, and timestamps.
# MAGIC
# MAGIC Output:
# MAGIC - Raw CSV at `dbfs:/tmp/spark_tutorial/raw/sales.csv`

# COMMAND ----------

# MAGIC %python
from pyspark.sql import functions as F, types as T
import random

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

# Create a synthetic DataFrame
n = rows
categories = ["Books","Games","Electronics","Grocery","Clothing"]
countries  = ["CA","US","UK","DE","FR","JP","CN"]

df = (
    spark.range(0, n)
    .withColumn("order_id", (F.col("id")/10).cast("long"))
    .withColumn("item_id", F.col("id"))
    .withColumn("category", F.element_at(F.array(*[F.lit(c) for c in categories]), (F.col("id") % len(categories)) + 1))
    .withColumn("country",  F.element_at(F.array(*[F.lit(c) for c in countries]), (F.col("id") % len(countries)) + 1))
    .withColumn("price",    (F.rand(seed=42) * 90 + 10).cast("double"))  # 10–100
    .withColumn("quantity", (F.rand(seed=7) * 4 + 1).cast("int"))        # 1–5
    .withColumn("ts",       (F.current_timestamp() - F.expr("INTERVAL int(id%72) HOURS")))
    .drop("id")
)

# Write out as raw CSV (partitioned by date for realism)
(df
 .withColumn("ds", F.to_date("ts"))
 .write.mode("overwrite")
 .option("header", True)
 .partitionBy("ds")
 .csv(f"{raw_path}/sales_csv")
)
print("Wrote CSV to:", f"{raw_path}/sales_csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Load Data with a Schema
# MAGIC We'll read the CSV back with an explicit schema (good practice for production).

# COMMAND ----------

# MAGIC %python
schema = T.StructType([
    T.StructField("order_id",  T.LongType(),   True),
    T.StructField("item_id",   T.LongType(),   True),
    T.StructField("category",  T.StringType(), True),
    T.StructField("country",   T.StringType(), True),
    T.StructField("price",     T.DoubleType(), True),
    T.StructField("quantity",  T.IntegerType(),True),
    T.StructField("ts",        T.TimestampType(), True),
])

sales_df = (
    spark.read
    .schema(schema)
    .option("header", True)
    .csv(f"{raw_path}/sales_csv")
)

display(sales_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) DataFrame API Essentials
# MAGIC - Select & filter
# MAGIC - Derived columns
# MAGIC - Grouped aggregations
# MAGIC - Explain plans & caching

# COMMAND ----------

# MAGIC %python
from pyspark.sql import functions as F

# Basic projections and filters
basic_df = sales_df.select("order_id", "item_id", "category", "country", "price", "quantity", "ts")                    .where((F.col("price") > 50) & (F.col("country").isin("CA","US")))

# Derived columns
enriched_df = basic_df.withColumn("revenue", F.col("price") * F.col("quantity"))                       .withColumn("hour", F.hour("ts"))

# Aggregation
agg_df = (
    enriched_df.groupBy("country","category")
    .agg(F.countDistinct("order_id").alias("orders"),
         F.sum("quantity").alias("units"),
         F.round(F.sum("revenue"), 2).alias("revenue"))
    .orderBy(F.desc("revenue"))
)

display(agg_df)

# Cache hot data
enriched_df.cache().count()
print("Cached enriched_df.")

# Explain a query plan
agg_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Window Functions
# MAGIC Compute per-country revenue ranks by category.

# COMMAND ----------

# MAGIC %python
from pyspark.sql.window import Window as W

w = W.partitionBy("country").orderBy(F.desc("revenue"))
ranked_df = agg_df.withColumn("rev_rank_in_country", F.dense_rank().over(w))

display(ranked_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Spark SQL
# MAGIC Register a temp view and run SQL. Equivalent to the DataFrame API above.

# COMMAND ----------

# MAGIC %python
enriched_df.createOrReplaceTempView("sales_enriched")

# COMMAND ----------

# MAGIC %sql
-- Top revenue categories by country (SQL version)
SELECT country, category, COUNT(DISTINCT order_id) AS orders,
       SUM(quantity) AS units,
       ROUND(SUM(price*quantity), 2) AS revenue
FROM sales_enriched
GROUP BY country, category
ORDER BY revenue DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Joins
# MAGIC We'll create a small dimension table and join.

# COMMAND ----------

# MAGIC %python
category_dim = spark.createDataFrame(
    [("Books", "Media"), ("Games","Media"), ("Electronics","Hardgoods"),
     ("Grocery","Consumable"), ("Clothing","Apparel")],
    ["category","dept"]
)

joined = enriched_df.join(category_dim, on="category", how="left")
display(joined.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) User-Defined Functions (UDF)
# MAGIC UDFs let you extend Spark. (Use sparingly; prefer built-ins for speed.)

# COMMAND ----------

# MAGIC %python
from pyspark.sql.functions import udf

@udf("string")
def price_band(p: float) -> str:
    if p is None: return "unknown"
    return "high" if p >= 75 else "mid" if p >= 40 else "low"

with_bands = enriched_df.withColumn("price_band", price_band(F.col("price")))
display(with_bands.groupBy("price_band").agg(F.count("*").alias("cnt")).orderBy("price_band"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8) Spark ML: Build a Simple Model
# MAGIC We'll predict `quantity` from `price` and hour-of-day.
# MAGIC
# MAGIC - VectorAssembler
# MAGIC - Linear Regression
# MAGIC - Pipeline + Train/Test split

# COMMAND ----------

# MAGIC %python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

ml_df = enriched_df.select("quantity","price","hour").na.drop()
assembler = VectorAssembler(inputCols=["price","hour"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="quantity")
pipeline = Pipeline(stages=[assembler, lr])

train, test = ml_df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
pred = model.transform(test)

display(pred.select("price","hour","quantity","prediction").limit(20))

lr_model = model.stages[-1]
print("Coefficients:", lr_model.coefficients, "Intercept:", lr_model.intercept)
print("RMSE:", lr_model.summary.rootMeanSquaredError, "R2:", lr_model.summary.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9) Delta Lake: Bronze/Silver/Gold Sketch
# MAGIC We'll write our enriched data as a **Delta** table and demonstrate **MERGE** and **Time Travel**.

# COMMAND ----------

# MAGIC %python
# Write Delta
(enriched_df
 .write
 .format("delta")
 .mode("overwrite")
 .save(f"{silver_path}/sales_enriched_delta")
)

spark.sql(f"CREATE TABLE IF NOT EXISTS {full_table_name} USING DELTA LOCATION '{silver_path}/sales_enriched_delta'")
display(spark.sql(f"SELECT * FROM {full_table_name} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### MERGE (Upsert) Example
# MAGIC We'll simulate late-arriving data: change a couple of rows and MERGE into the Delta table.

# COMMAND ----------

# MAGIC %python
updates = enriched_df.limit(2).withColumn("quantity", F.col("quantity") + 10)
updates.createOrReplaceTempView("updates")

spark.sql(f"""
MERGE INTO {full_table_name} AS t
USING updates AS s
ON t.item_id = s.item_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql(f"SELECT * FROM {full_table_name} ORDER BY ts DESC LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Travel
# MAGIC Query an earlier version of the Delta table.

# COMMAND ----------

# MAGIC %python
history = spark.sql(f"DESCRIBE HISTORY {full_table_name}")
display(history)

# Pick version 0 if present
min_ver = history.agg(F.min("version")).first()[0]
tt = spark.read.format("delta").option("versionAsOf", int(min_ver)).load(f"{silver_path}/sales_enriched_delta")
print("Time-traveled to version:", int(min_ver))
display(tt.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10) Structured Streaming (Mini Demo)
# MAGIC We'll use the built-in **rate** source to generate rows and compute a moving sum.

# COMMAND ----------

# MAGIC %python
stream_path = f"{base_path}/stream_demo"

# Clean any prior run
dbutils.fs.rm(stream_path, True)

stream_df = (spark.readStream
             .format("rate")
             .option("rowsPerSecond", 5)
             .load()
             .withColumn("value", (F.col("value") % 10).cast("int"))
            )

agg_stream = (stream_df
              .groupBy(F.window("timestamp","10 seconds"), F.col("value"))
              .agg(F.count("*").alias("cnt"))
             )

q = (agg_stream
     .writeStream
     .format("delta")
     .outputMode("complete")
     .option("checkpointLocation", f"{stream_path}/chk")
     .option("path", f"{stream_path}/out")
     .trigger(processingTime="10 seconds")
     .start()
    )

print("Streaming started. Let it run for ~20–30 seconds, then stop the stream in the cell output or with q.stop().")

# COMMAND ----------

# MAGIC %md
# MAGIC After ~20–30 seconds, read the streaming output (Delta files) as a static table.

# COMMAND ----------

# MAGIC %python
# Stop the stream if it's still running
try:
    if q.isActive:
        q.stop()
except NameError:
    pass

stream_out = spark.read.format("delta").load(f"{stream_path}/out")
display(stream_out.orderBy("window"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11) Performance Tips (Quick Hits)
# MAGIC - **Prefer built-in functions** over UDFs; use **pandas UDFs** if needed.
# MAGIC - **Specify schemas** for file reads.
# MAGIC - **Cache sparingly** (only hot intermediate results).
# MAGIC - **Use Delta** for reliable, fast lakehouse storage.
# MAGIC - **Partition wisely** by low-cardinality columns you filter on often.
# MAGIC - **Broadcast joins** when one side is small: `broadcast(df)`.
# MAGIC - Inspect plans with `.explain()` and leverage AQE and adaptive joins.
# MAGIC - For production, use **Workflows/Jobs** to schedule and monitor.
# MAGIC
# MAGIC ## 12) Cleanup
# MAGIC Remove tutorial data & tables when done.

# COMMAND ----------

# MAGIC %python
# Uncomment to clean up:
# spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
# dbutils.fs.rm(base_path, True)
# print("Cleaned tutorial artifacts.")
