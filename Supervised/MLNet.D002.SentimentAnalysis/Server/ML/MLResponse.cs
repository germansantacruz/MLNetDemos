using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLNet.D002.SentimentAnalysis.Server.ML
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
