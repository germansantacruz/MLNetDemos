using MLNet.D006.BinaryClassification.ML;
using System;

namespace MLNet.D006.BinaryClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("************************************************************");
            Console.WriteLine("ML Supervisado");
            Console.WriteLine("Problema: Clasificación binaria");
            Console.WriteLine("Algoritmo: FastTree");
            Console.WriteLine("BinaryClassification.Trainers.FastTree\n");
            Console.WriteLine("Desc.: Predecir si el precio de un auto es un buen trato o no.");
            Console.WriteLine("************************************************************\n");
                  
            /*
            Trainer trainer = new Trainer();
            trainer.Train();*/
                        
            Predictor predictor = new Predictor();
            predictor.Predict("input.json");

            Console.ReadLine();
        }
    }
}
