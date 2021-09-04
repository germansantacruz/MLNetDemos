using Microsoft.ML;
using MLNet.Helpers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML.Supervised.SimpleRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Preparar los datos
            string path = Path.Combine(Environment.CurrentDirectory, "Data", "salarios.csv");
            MLContext mlContext = new MLContext();             
            IDataView splitDataView = MLHelper.LoadDataFromTextFile<ModelInput>(mlContext, path, hasHeader: true, separatorChar: '|');

            // 2. Construir y entrenar el modelo           
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView);

            // 3. Evaluar el modelo
            //Evaluate(mlContext, model, splitDataView.TestSet);

            // 4. Usar el modelo 
            UseModelWithSingleItem(mlContext, model, 1f);
            UseModelWithSingleItem(mlContext, model, 3f);
            UseModelWithSingleItem(mlContext, model, 5f);
            UseModelWithSingleItem(mlContext, model, 7f);
            UseModelWithSingleItem(mlContext, model, 10f);

            Console.ReadLine();
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Concatenate(outputColumnName: "Features", "YearsOfExperience")
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Salary", featureColumnName: "Features", maximumNumberOfIterations: 100));
                
                /*
                Concatenate(outputColumnName: "Features",
                inputColumnNames: nameof(ModelInput.YearsOfExperience))                                    
                                    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features", maximumNumberOfIterations: 100));
                */
            Console.WriteLine("=============== Crear y entrenar el modelo ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== Fin del entrenamiento ===============");
            Console.WriteLine();

            return model;
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, float yearsOfExperience)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            ModelInput sample = new ModelInput
            {
                YearsOfExperience = yearsOfExperience
            };           

            var resultPrediction = predictionFunction.Predict(sample);           
            Console.WriteLine($"Years: {yearsOfExperience} - Salario predicción: {resultPrediction.Salary}");            
        }
    }
}
