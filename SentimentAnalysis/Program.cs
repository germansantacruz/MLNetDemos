using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

        static void Main(string[] args)
        {
            /* Preparar los datos
             * 1. Copiar el archivo al proyecto y en su propiedad "Copy to Output Directory"
             *    setear a "Copy if newer"
             * 2. Crear clases para sus datos de entrada y predicciones:
             *    SentimentData y SentimentPrediction
             */

            /* Cargar los datos y dividir en 2 sets: Train y Test
             * Los datos pueden ser cargados de un archivo de texto o base de datos como
             * SQL Server, archivos log a un objeto IDataView
             */
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);

            /* Build y entrenar el modelo
             * Extraer y transformar los datos
             * Entrenar el modelo             * 
             */
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            // Evaluar el modelo
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithSingleItem(mlContext, model);
            UseModelWithBatchItems(mlContext, model);

            Console.ReadLine();
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
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

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            // Transform() hace predicciones para múltiples filas de entrada
            IDataView predictions = model.Transform(splitTestSet);
            // el método Evaluate () evalúa el modelo, que compara los valores predichos con las
            // etiquetas reales en el conjunto de datos de prueba y devuelve un objeto
            // CalibratedBinaryClassificationMetrics sobre el rendimiento del modelo. 
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            // obtiene la precisión de un modelo, que es la proporción de predicciones correctas
            // en el conjunto de prueba
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            // indica la confianza que tiene el modelo al clasificar correctamente las clases positivas y negativas.
            // Quieres que AreaUnderRocCurve esté lo más cerca posible de uno
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            // La métrica F1Score obtiene la puntuación F1 del modelo, que es una medida del equilibrio entre precisión y recuperación.
            // Desea que F1Score sea lo más cercano posible a uno.
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");                       
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            // PredictionEngine is not thread-safe. It's acceptable to use in single-threaded or
            // prototype environments. For improved performance and thread safety in production
            // environments, use the PredictionEnginePool service
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }

        /*
         *  Si no está satisfecho con la calidad del modelo, puede intentar mejorarlo
         *  proporcionando conjuntos de datos de entrenamiento más grandes o eligiendo 
         *  diferentes algoritmos de entrenamiento con diferentes 
         *  hiperparámetros para cada algoritmo. 
         * */
    }
}
