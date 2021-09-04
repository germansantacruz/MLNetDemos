using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLNet.D003.SentimentAnalysis.ML
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
            var trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(dataPath, separatorChar: ',', allowQuoting: true);           
            var splitDataView = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            // Transformar datos
            var booleanMap = new[] {
                new KeyValuePair<int, bool>(0, false),
                new KeyValuePair<int, bool>(2, true),
                new KeyValuePair<int, bool>(4, true)
            };
            
            var transformLabel = mlContext.Transforms.Conversion.MapValue(
                outputColumnName: "LabelBool",
                booleanMap,
                inputColumnName: nameof(ModelInput.Label)
            );          

            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(ModelInput.Text))
                .Append(transformLabel);
                        
            var sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "LabelBool",
                featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);          

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(splitDataView.TrainSet);
            mlContext.Model.Save(trainedModel, splitDataView.TrainSet.Schema, modelPath);

            // Evaluar el modelo con los datos Test
            var testSetTransform = trainedModel.Transform(splitDataView.TestSet);

            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: "LabelBool",
                scoreColumnName: nameof(ModelOuput.Score));

            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}\n" +
                              $"AUC: {modelMetrics.AreaUnderRocCurve:P2}\n" +
                              $"F1Score: {modelMetrics.F1Score:P2}\n" +
                              $"Positive Recall: {modelMetrics.PositiveRecall}\n" +
                              $"Negative Recall: {modelMetrics.NegativeRecall}\n\n");
        }
    }
}
