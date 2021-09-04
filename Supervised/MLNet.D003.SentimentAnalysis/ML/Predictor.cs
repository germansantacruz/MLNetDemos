using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D003.SentimentAnalysis.ML
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

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOuput>(mlModel);
            var resultPrediction = predictionEngine.Predict(new ModelInput { Text = inputData });
            string sentiment = string.Empty;

            switch (resultPrediction.Prediction)
            {
                case false:
                    sentiment = "Negative";
                    break;
                case true:
                    sentiment = "Positive";
                    break;               
            }

            Console.WriteLine($"Texto: {inputData}\n" +
                              $"Predicción: {sentiment}\n" +
                              $"Probabilidad: {resultPrediction.Probability:P0}\n\n");
        }
    }
}
