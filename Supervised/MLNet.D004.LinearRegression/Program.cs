using MLNet.D004.LinearRegression.ML;
using System;

namespace MLNet.D004.LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {    
            /*
            Trainer trainer = new Trainer();
            trainer.Train();*/
            
            Predictor predictor = new Predictor();
            predictor.Predict("input.json");
            
            Console.ReadLine();
        }
    }
}
