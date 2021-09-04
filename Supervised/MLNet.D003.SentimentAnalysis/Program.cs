using MLNet.D003.SentimentAnalysis.ML;
using System;

namespace MLNet.D003.SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            Trainer trainer = new Trainer();
            trainer.Train();*/

            Predictor predictor = new Predictor();
            predictor.Predict(@"@stellargirl I loooooooovvvvvveee my Kindle2. Not that the DX is cool, but the 2 is fantastic in its own right.");

            Console.ReadLine();
        }
    }
}
