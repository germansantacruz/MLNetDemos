using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D003.SentimentAnalysis.ML
{
    public class BaseML
    {       
        static readonly string DATA_FILENAME = "training.1600000.processed.noemoticon.csv";
        static readonly string MODEL_FILENAME = "demo003.zip";
     
        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);
        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Models", MODEL_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext();
        }
    }
}
