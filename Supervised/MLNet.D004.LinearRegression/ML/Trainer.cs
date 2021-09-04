using Microsoft.ML;
using MLNet.D004.LinearRegression.Common;
using System;
using System.IO;
using System.Linq;

namespace MLNet.D004.LinearRegression.ML
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
            var trainingDataView = mlContext.Data.LoadFromTextFile<EmploymentHistory>(dataPath, ',');
            var splitDataView = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.4);

            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(EmploymentHistory.DurationInMonths))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.IsMarried)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.BSDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.MSDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.YearsExperience))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.AgeAtHire)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.HasKids)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.WithinMonthOfVesting)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.DeskDecorations)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.LongCommute)))
                .Append(mlContext.Transforms.Concatenate("Features",
                    typeof(EmploymentHistory).ToPropertyList<EmploymentHistory>(nameof(EmploymentHistory.DurationInMonths)))));

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(splitDataView.TrainSet);
            mlContext.Model.Save(trainedModel, splitDataView.TrainSet.Schema, modelPath);

            // Evaluar el modelo con los datos Test
            var testSetTransform = trainedModel.Transform(splitDataView.TestSet);
            var modelMetrics = mlContext.Regression.Evaluate(testSetTransform);

            Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}\n" +
                              $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}\n" +
                              $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}\n" +
                              $"RSquared: {modelMetrics.RSquared:0.##}\n" +
                              $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");
        }
    }
}
