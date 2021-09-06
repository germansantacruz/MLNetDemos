using MLNet.D004.LinearRegression.ML;
using System;

namespace MLNet.D004.LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("************************************************************");
            Console.WriteLine("ML Supervisado");
            Console.WriteLine("Problema: Regresión");
            Console.WriteLine("Algoritmo: Regresión lineal múltiple");
            Console.WriteLine("Regression.Trainers.Sdca\n");
            Console.WriteLine("Desc.: Predecir la deserción de los empleados (en meses) en\nfunción de varios atributos de los empleados.");
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
