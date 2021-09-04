using Microsoft.ML.Data;

namespace MLNet.D001.SentimentAnalysis.ML
{
    public class RestaurantPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
