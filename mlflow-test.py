from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import mlflow

import pandas as pd
import os


## 환경변수 설정
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.14:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.0.14:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

## model-name, hyperparameter 입력 using argparse
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model-name", dest="model_name", type=str, default="random-forest")
parser.add_argument("--maxDepth", dest="maxDepth", type=int, default=5)
args = parser.parse_args()

## experiment 설정
mlflow.set_experiment("BOAZ-Test")

# SparkSession 생성
spark = (SparkSession
         .builder
         .appName(f"{args.model_name}")
         .getOrCreate())

## Spark Pipeline 생성
filePath = "./data/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed = 42)
categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]
stringIndexer = StringIndexer(inputCols = categoricalCols,
                              outputCols = indexOutputCols,
                              handleInvalid = "skip")
numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols = assemblerInputs,
                               outputCol = "features")
rf = RandomForestRegressor(labelCol = "price", maxBins = 40, maxDepth = args.maxDepth, numTrees = 10, seed = 42)
pipeline = Pipeline(stages = [stringIndexer, vecAssembler, rf])



## 실행
with mlflow.start_run():
    
    mlflow.log_param("num_trees", rf.getNumTrees())
    mlflow.log_param("max_depth", rf.getMaxDepth())

    pipelineModel = pipeline.fit(trainDF)
    mlflow.spark.log_model(pipelineModel,
                           args.model_name)

    predDF = pipelineModel.transform(testDF)
    regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")
    rmse = regressionEvaluator.setMetricName("rmse").evaluate(predDF)
    r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    
spark.stop()