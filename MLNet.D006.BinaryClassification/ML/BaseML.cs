using Microsoft.ML;
using System;
using System.IO;

namespace MLNet.D006.BinaryClassification.ML
{
    public class BaseML
    {
        static readonly string DATA_FILENAME = "sampledata.csv";
        static readonly string TEST_DATA_FILENAME = "testdata.csv";
        static readonly string MODEL_FILENAME = "demo006.zip";

        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);
        protected readonly string testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", TEST_DATA_FILENAME);
        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Models", MODEL_FILENAME);
        
        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext(2020);
        }
    }
}
