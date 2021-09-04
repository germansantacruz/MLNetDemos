using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace MLNet.D001.SentimentAnalysis.ML
{
    public class Trainer : BaseML
    {
        public void Train()
        {
            if (!File.Exists(dataPath))
            {                
                Console.WriteLine($"‎‎Error al buscar el archivo de datos de entrenamiento {dataPath}.");
                return;
            }

            // Cargar los datos
            var trainingDataView = mlContext.Data.LoadFromTextFile<RestaurantFeedback>(dataPath);
            var splitDataView = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(RestaurantFeedback.Text));

            var sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(RestaurantFeedback.Label),
                featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(splitDataView.TrainSet);
            mlContext.Model.Save(trainedModel, splitDataView.TrainSet.Schema, modelPath);

            // Evaluar el modelo con los datos Test
            var testSetTransform = trainedModel.Transform(splitDataView.TestSet);

            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: nameof(RestaurantFeedback.Label),
                scoreColumnName: nameof(RestaurantPrediction.Score));

            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}\n" +
                              $"AUC: {modelMetrics.AreaUnderRocCurve:P2}\n" +
                              $"F1Score: {modelMetrics.F1Score:P2}\n" +
                              $"Positive Recall: {modelMetrics.PositiveRecall}\n" +
                              $"Negative Recall: {modelMetrics.NegativeRecall}\n\n");
        }
    }
}
