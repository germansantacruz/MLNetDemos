using Microsoft.ML;
using MLNet.D002.SentimentAnalysis.Server.DataModels;
using System.IO;
using System.Linq;

namespace MLNet.D002.SentimentAnalysis.Server.ML
{
    public class Trainer : BaseML
    {
        public MLResponse TrainAndSaveModel()
        {
            var mlResponse = new MLResponse();
            
            if (!File.Exists(dataPath))
            {
                mlResponse.Message = $"‎‎Error al buscar el archivo de datos de entrenamiento {dataPath}.";
                mlResponse.Error = true;
                return mlResponse;
            }

            // Cargar los datos
            var trainingDataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, separatorChar: '|');
            
            // Construir el modelo
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SentimentData.SentimentText));

            var sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            // Entrenar y guardar el modelo
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);

            mlResponse.Error = false;
            mlResponse.Message = "Modelo entrenado y guardado correctamente.";

            return mlResponse;
        }
    }
}
