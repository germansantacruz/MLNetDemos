using Microsoft.ML.Data;

namespace MLNet.D002.SentimentAnalysis.Server.DataModels
{
    public class SentimentData
    {
        [LoadColumn(0), ColumnName("SentimentText")]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }
}
