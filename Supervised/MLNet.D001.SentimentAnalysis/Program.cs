using MLNet.D001.SentimentAnalysis.ML;
using System;
using System.IO;

namespace MLNet.D001.SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {           
            string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "sampledata.csv");

            Console.WriteLine($"Opciones: \n" +
                              $"-> train \n" +
                              $"-> predict <texto> \n" +
                              $"-> quit");
   
            string[] argsOpcion = new[] { "" };
            string input;

            while (!argsOpcion[0].Contains("quit"))
            {
                input = Console.ReadLine();
                argsOpcion = input.Split(' ');
                switch (argsOpcion[0])
                {
                    case "predict":
                        new Predictor().Predict(argsOpcion[1]);
                        break;
                    case "train":
                        new Trainer().Train();
                        break;
                    default:
                        Console.WriteLine($"argsOpcion[0] no es una opción válida.");
                        break;
                }
            }
        }
    }
}
