using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using MLNet.D002.SentimentAnalysis.Server.DataModels;
using MLNet.D002.SentimentAnalysis.Server.ML;
using System;
using System.Diagnostics;

namespace MLNet.D002.SentimentAnalysis.Server.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SentimentController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> _predictionEnginePool;
        private readonly Trainer _trainer;

        public SentimentController(PredictionEnginePool<SentimentData, SentimentPrediction> predictionEnginePool,
            Trainer trainer)
        {
            _predictionEnginePool = predictionEnginePool;
            _trainer = trainer;
        }

        [Route("predict")]
        [HttpGet]
        public ActionResult<float> PredictSentiment([FromQuery] string sentimentText)
        {
            var prediction = _predictionEnginePool.Predict(new SentimentData { SentimentText = sentimentText });
            Debug.WriteLine("---------------");
            Debug.WriteLine($"{sentimentText}");
            Debug.WriteLine($"{prediction.Score}");
            Debug.WriteLine($"{CalculatePercentage(prediction.Score)}");
            Debug.WriteLine("---------------");
            return CalculatePercentage(prediction.Score);
        }

        [Route("train")]
        [HttpGet]
        public ActionResult<MLResponse> TrainModel()
        {
            return _trainer.TrainAndSaveModel();
        }

        private float CalculatePercentage(double value)
        {
            return 100 * (1.0f / (1.0f + (float)Math.Exp(-value)));
        }
    }
}
