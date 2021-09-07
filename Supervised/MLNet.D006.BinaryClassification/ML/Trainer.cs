using Microsoft.ML;
using MLNet.D006.BinaryClassification.Common;
using System;
using System.IO;
using System.Linq;

namespace MLNet.D006.BinaryClassification.ML
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

            if (!File.Exists(testDataPath))
            {
                Console.WriteLine($"‎‎Error al buscar el archivo de datos de test {testDataPath}.");
                return;
            }

            // Cargar los datos
            var trainingDataView = mlContext.Data.LoadFromTextFile<CarInventory>(dataPath, ',', hasHeader: false);
            var testDataView = mlContext.Data.LoadFromTextFile<CarInventory>(testDataPath, ',', hasHeader: false);

            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                    typeof(CarInventory).ToPropertyList<CarInventory>(nameof(CarInventory.Label)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(
                    inputColumnName: "Features",
                    outputColumnName: "FeaturesNormalizedByMeanVar"));

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(CarInventory.Label),
                featureColumnName: "FeaturesNormalizedByMeanVar",
                numberOfLeaves: 2,
                numberOfTrees: 1000,
                minimumExampleCountPerLeaf: 1,
                learningRate: 0.2);
                        
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);

            // Evaluar el modelo con los datos Test
            /*var evaluationPipeline = trainedModel.Append(mlContext.Transforms
                .CalculateFeatureContribution(trainedModel.LastTransformer)
                .Fit(dataProcessPipeline.Fit(trainingDataView).Transform(trainingDataView)));*/

            var testSetTransform = trainedModel.Transform(testDataView);
            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: nameof(CarInventory.Label),
                scoreColumnName: "Score");

            Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:P2}");
            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"Area under Precision recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1Score: {modelMetrics.F1Score:P2}");
            Console.WriteLine($"LogLoss: {modelMetrics.LogLoss:#.##}");
            Console.WriteLine($"LogLossReduction: {modelMetrics.LogLossReduction:#.##}");
            Console.WriteLine($"PositivePrecision: {modelMetrics.PositivePrecision:#.##}");
            Console.WriteLine($"PositiveRecall: {modelMetrics.PositiveRecall:#.##}");
            Console.WriteLine($"NegativePrecision: {modelMetrics.NegativePrecision:#.##}");
            Console.WriteLine($"NegativeRecall: {modelMetrics.NegativeRecall:P2}");
        }
    }
}
