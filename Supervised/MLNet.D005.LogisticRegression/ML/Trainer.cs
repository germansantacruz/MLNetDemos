using Microsoft.ML;
using MLNet.D005.LogisticRegression.Common;
using System;
using System.IO;
using System.Linq;

namespace MLNet.D005.LogisticRegression.ML
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
            var trainingDataView = mlContext.Data.LoadFromTextFile<FileInput>(dataPath);
            var splitDataView = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(FileInput.Strings));
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(FileInput.Label), 
                featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(splitDataView.TrainSet);
            mlContext.Model.Save(trainedModel, splitDataView.TrainSet.Schema, modelPath);

            // Evaluar el modelo con los datos Test
            var testSetTransform = trainedModel.Transform(splitDataView.TestSet);

            /*
            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: nameof(FileInput.Label),
                scoreColumnName: nameof(FilePrediction.Score));

            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}\n" +
                              $"AUC: {modelMetrics?.AreaUnderRocCurve:P2}\n" +
                              $"F1Score: {modelMetrics.F1Score:P2}\n" +
                              $"Positive Recall: {modelMetrics?.PositiveRecall}\n" +
                              $"Negative Recall: {modelMetrics?.NegativeRecall}\n\n");*/
        }
    }
}
