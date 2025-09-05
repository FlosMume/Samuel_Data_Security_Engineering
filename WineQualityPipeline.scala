// ===================================================================================
// FILE: WineQualityPipeline.scala
// DESCRIPTION: Same pipeline in Scala (more performant, type-safe)
// ===================================================================================

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

object WineQualityPipeline {
  def main(args: Array[String]): Unit = {

    // Step 1: Create Spark Session
    val spark = SparkSession.builder()
      .appName("WineQualityPrediction")
      .config("spark.sql.adaptive.enabled", "true")
      .getOrCreate()

    import spark.implicits._

    // Step 2: Load Data
    val df = spark.read
      .option("sep", ";")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("winequality-red.csv")

    df.printSchema()
    df.show(5)

    // Step 3: Clean Data using SQL
    df.createOrReplaceTempView("wine_data")
    val cleanDF = spark.sql("""
      SELECT *
      FROM wine_data
      WHERE quality IS NOT NULL
        AND alcohol > 0 AND pH > 0
    """)

    // Step 4: Feature Engineering
    val featureColumns = cleanDF.columns.filter(_ != "quality")

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("raw_features")
      .setHandleInvalid("skip")

    // Step 5: Scale Features
    val scaler = new StandardScaler()
      .setInputCol("raw_features")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    // Step 6: Define Model
    val lr = new LinearRegression()
      .setLabelCol("quality")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(100)

    // Step 7: Build Pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, scaler, lr))

    // Step 8: Split Data
    val Array(trainData, testData) = cleanDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // Step 9: Hyperparameter Tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val evaluator = new RegressionEvaluator()
      .setLabelCol("quality")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(3)

    // Step 10: Train
    val model = cv.fit(trainData)

    // Step 11: Predict
    val predictions = model.transform(testData)
    predictions.select("quality", "prediction").show(10)

    // Step 12: Evaluate
    val rmse = evaluator.evaluate(predictions)
    println(s"RMSE: $rmse")

    val r2Evaluator = evaluator.setMetricName("r2")
    val r2 = r2Evaluator.evaluate(predictions)
    println(s"R²: $r2")

    // Step 13: Save Model (Optional)
    // model.bestModel.write.overwrite().save("wine_quality_model_scala")

    // Step 14: Stop
    spark.stop()
  }
}