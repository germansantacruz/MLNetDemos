namespace MLNet.D002.SentimentAnalysis.Client
{
    public class MLResponse
    {
        public bool Error { get; set; }
        public string Message { get; set; }

        public MLResponse()
        {
            Error = false;
            Message = "";
        }
    }
}
