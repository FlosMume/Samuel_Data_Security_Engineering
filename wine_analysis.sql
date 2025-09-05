-- File: wine_analysis.sql
-- Run via Spark SQL or spark.sql()

-- Summary statistics
SELECT 
  COUNT(*) AS total_rows,
  AVG(quality) AS avg_quality,
  MIN(quality) AS min_quality,
  MAX(quality) AS max_quality
FROM wine_data;

-- Correlation between alcohol and quality
SELECT 
  CORR(alcohol, quality) AS alcohol_quality_corr
FROM wine_data;

-- Average quality by alcohol level (binned)
SELECT 
  FLOOR(alcohol) AS alcohol_bin,
  AVG(quality) AS avg_quality,
  COUNT(*) AS count_wines
FROM wine_data
GROUP BY FLOOR(alcohol)
ORDER BY alcohol_bin;

-- Wines with high citric acid and low volatile acidity
SELECT *
FROM wine_data
WHERE citric_acid > 0.5 AND volatile_acidity < 0.4
ORDER BY quality DESC;