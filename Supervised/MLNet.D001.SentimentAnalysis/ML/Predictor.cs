using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D001.SentimentAnalysis.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputData)
        {
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Error al buscar el modelo en {modelPath}.");
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

            var predictionEngine = mlContext.Model.CreatePredictionEngine<RestaurantFeedback, RestaurantPrediction>(mlModel);
            var resultPrediction = predictionEngine.Predict(new RestaurantFeedback { Text = inputData });

            Console.WriteLine($"Texto: {inputData}\n" +
                              $"Predicción: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Negative" : "Positive")}\n" +
                              $"Probabilidad: {resultPrediction.Probability:P0}\n\n");
        }
    }
}
