using MLNet.D005.LogisticRegression.ML;
using System;

namespace MLNet.D005.LogisticRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("************************************************************");
            Console.WriteLine("ML Supervisado");
            Console.WriteLine("Problema: Clasificación");
            Console.WriteLine("Algoritmo: Regresión Logística Binaria\n");
            Console.WriteLine("Desc.: Análisis básico de archivos estáticos para determinar,\nsi un archivo es malicioso o benigno.");
            Console.WriteLine("************************************************************\n");

            /*
            var extractor = new FeatureExtractor();
            extractor.Extract("temp_data");*/

            /*
            Trainer trainer = new Trainer();
            trainer.Train();*/

            Predictor predictor = new Predictor();
            predictor.Predict("MLNet.D005.LogisticRegression.exe");

            Console.ReadLine();
        }
    }
}
