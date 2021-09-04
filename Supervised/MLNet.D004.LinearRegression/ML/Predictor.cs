using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.IO;

namespace MLNet.D004.LinearRegression.ML
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

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EmploymentHistory, EmploymentHistoryPrediction>(mlModel);
            var json = File.ReadAllText(inputDataFile);
            var prediction = predictionEngine.Predict(JsonConvert.DeserializeObject<EmploymentHistory>(json));

            Console.WriteLine($"Based on input json:\n" +
                              $"{json}\n" +
                              $"The employee is predicted to work {prediction.DurationInMonths:#.##} months");
        }
    }
}
