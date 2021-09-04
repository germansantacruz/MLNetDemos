using Microsoft.ML.Data;

namespace MLNet.D003.SentimentAnalysis.ML
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public int Label { get; set; }

        [LoadColumn(5)]
        public string Text { get; set; }
    }
}
