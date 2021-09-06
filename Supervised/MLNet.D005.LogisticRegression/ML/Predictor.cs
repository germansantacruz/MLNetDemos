using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.IO;

namespace MLNet.D005.LogisticRegression.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputDataFile)
        {
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Error al buscar el modelo en {modelPath}.");
                return;
            }

            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Error al buscar el archivo de datos de entrada en {inputDataFile}");
                return;
            }

            // Cargar el modelo
            ITransformer mlModel;

            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);               
            }

            if (mlModel == null)
            {
                Console.WriteLine("Error al cargar el modelo.");
                return;
            }    

            var predictionEngine = mlContext.Model.CreatePredictionEngine<FileInput, FilePrediction>(mlModel);
            var prediction = predictionEngine.Predict(new FileInput() { 
                Strings = GetStrings(File.ReadAllBytes(inputDataFile))
            });

            Console.WriteLine($"Basado en el análisis del archivo ({inputDataFile}), se clasifica como: {(prediction.IsMalicious ? "malicioso" : "benigno")}" +
                              $" con un nivel de confianza de {prediction.Probability:P0}");
        }
    }
}
