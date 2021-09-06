using Microsoft.ML.Data;

namespace MLNet.D005.LogisticRegression.ML
{
    public class FilePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsMalicious { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
