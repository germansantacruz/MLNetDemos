using Microsoft.ML;
using System;
using System.IO;

namespace Demo001
{
    class Program
    {
        static readonly string TrainDataPath = Path.Combine(Environment.CurrentDirectory,
            "Data", "AgeRangeData03.csv");

        static void Main(string[] args)
        {
            EntrenamientoConMasFeatures();
            Console.ReadLine();
        }

        private static void PredictSimple(string name, float age, string gender, PredictionEngine<AgeRange, AgeRangePrediction> predictionFunction)
        {
            var example = new AgeRange()
            {
                Age = age,
                Name = name,
                Gender = gender
            };

            var prediction = predictionFunction.Predict(example);
            Console.WriteLine($"Name: {example.Name}\t Age: {example.Age:00}\t Gender: {example.Gender}\t >> Predicted Label: {prediction.Label}");
        }

        private static void EntrenamientoConMasFeatures()
        {
            // Paso 1. Cargar los datos o informacion en un contexto de ML
            var ml = new MLContext(1);
            var data = ml.Data.LoadFromTextFile<AgeRange>(TrainDataPath, hasHeader: true,
                separatorChar: ',');

            // Paso 2. Entrenar el modelo
            // -- Se convierte la columna Label a un valor numerico Key
            // -- Luego con las caracteristicas que se va a trabajar como es nro no hay que hacer nada
            // -- Elegir el algoritmo de prediccion
            // -- Convertir el nro de la prediccion a un label
            var pipeline = ml.Transforms.Conversion.MapValueToKey("Label")
                .Append(ml.Transforms.Text.FeaturizeText("GenderFeat", "Gender"))
                .Append(ml.Transforms.Concatenate("Features", "Age", "GenderFeat"))
                .AppendCacheCheckpoint(ml)
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
            Console.WriteLine("Modelo entrenado");

            // Paso 3: predicciones
            var engine = ml.Model.CreatePredictionEngine<AgeRange, AgeRangePrediction>(model);
            PredictSimple("Jeff", 2, "M", engine);
            PredictSimple("Maria", 22, "F", engine);
            PredictSimple("German", 41, "M", engine);
            PredictSimple("Shelley", 9, "F", engine);
            PredictSimple("Jackie", 3, "F", engine);
            PredictSimple("Marco", 51, "M", engine);
            PredictSimple("Jim", 5, "M", engine);
        }

        private static void EntrenamientoSimple()
        {
            // Paso 1. Cargar los datos o informacion en un contexto de ML
            var ml = new MLContext(1);
            var data = ml.Data.LoadFromTextFile<AgeRange>(TrainDataPath, hasHeader: true,
                separatorChar: ',');

            // Paso 2. Entrenar el modelo
            // -- Se convierte la columna Label a un valor numerico Key
            // -- Luego con las caracteristicas que se va a trabajar como es nro no hay que hacer nada
            // -- Elegir el algoritmo de prediccion
            // -- Convertir el nro de la prediccion a un label
            var pipeline = ml.Transforms.Conversion.MapValueToKey("Label")
                .Append(ml.Transforms.Concatenate("Features", "Age"))
                .AppendCacheCheckpoint(ml)
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
            Console.WriteLine("Modelo entrenado");

            // Paso 3: predicciones
            var engine = ml.Model.CreatePredictionEngine<AgeRange, AgeRangePrediction>(model);
            PredictSimple("Jeff", 2, "M", engine);
            PredictSimple("Shelley", 9, "F", engine);
            PredictSimple("Jackie", 3, "F", engine);
            PredictSimple("Jim", 5, "M", engine);
        }


    }
}
