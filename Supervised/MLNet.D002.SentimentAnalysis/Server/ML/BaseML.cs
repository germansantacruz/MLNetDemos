using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D002.SentimentAnalysis.Server.ML
{
    public class BaseML
    {
        static readonly string MODEL_FILENAME = "sentimentAnalysis02.zip";
        static readonly string DATA_FILENAME = "my_custom_data.txt";

        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Models", MODEL_FILENAME);
        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext();
        }
    }
}
