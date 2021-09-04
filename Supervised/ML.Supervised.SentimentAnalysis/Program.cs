using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Helpers;
using System;
using System.Collections.Generic;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML.Supervised.SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            
            //my_custom_data yelp_labelled
            // 1. Preparar los datos
            string path = Path.Combine(Environment.CurrentDirectory, "Data", "my_custom_data.txt");
            MLContext mlContext = new MLContext();
            
            //TrainTestData splitDataView = MLHelper.LoadDataFromTextFile<SentimentData>(mlContext, path, separatorChar: '|', testFraction: 0.2);
            var splitDataView = MLHelper.LoadDataFromTextFile<SentimentData>(mlContext, path, separatorChar: '|');

            // 2. Construir y entrenar el modelo           
            //ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView);

            // Guardar el modelo
            mlContext.Model.Save(model, splitDataView.Schema, "sentimentAnalysis.zip");
            
            //ITransformer model = mlContext.Model.Load("sentimentAnalysis.zip", out var modelInputScheme);

            // 3. Evaluar el modelo
            //Evaluate(mlContext, model, splitDataView.TestSet);

            // 4. Usar el modelo 
            UseModelWithSingleItem(mlContext, model, "muy buen tutorial");
            UseModelWithBatchItems(mlContext, model);           


            Console.ReadLine();
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {           
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Crear y entrenar el modelo ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== Fin del entrenamiento ===============");
            Console.WriteLine();

            return model;
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, string sentimentText)
        {            
            var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = sentimentText
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prueba de Predicción del modelo con solo un texto ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine();
            Console.WriteLine("=============== Fin de la prueba de predicción ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "El articulo tenía muchos errores en el código."
                },
                new SentimentData
                {
                    SentimentText = "Muy buen post."
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = model.Transform(batchComments);            
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prueba de Predicción del modelo con varios datos ===============");
            Console.WriteLine();
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine();
            Console.WriteLine("=============== Fin de la prueba de predicción ===============");
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluando la precisión del modelo con Test data ===============");
            
            IDataView predictions = model.Transform(splitTestSet); 
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Métricas de calidad del modelo");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== Fin de la evalucación del modelo ===============");
        }        
    }
}
