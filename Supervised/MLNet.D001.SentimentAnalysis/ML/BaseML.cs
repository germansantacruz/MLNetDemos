using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D001.SentimentAnalysis.ML
{
    public class BaseML
    {
        static readonly string MODEL_FILENAME = "demo001.mdl";
        static readonly string DATA_FILENAME = "sampledata.csv";

        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, MODEL_FILENAME);
        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {            
            mlContext = new MLContext();
        }
    }
}
