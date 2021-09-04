using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D004.LinearRegression.ML
{
    public class BaseML
    {
        static readonly string DATA_FILENAME = "sampledata.csv";
        static readonly string MODEL_FILENAME = "demo004.zip";

        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);
        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Models", MODEL_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext(2020);
        }
    }
}
