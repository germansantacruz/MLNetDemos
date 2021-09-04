using Microsoft.ML.Data;

namespace MLNet.D003.SentimentAnalysis.ML
{
    public class ModelOuput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
