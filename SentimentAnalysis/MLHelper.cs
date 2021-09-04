using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML.Supervised.Helpers
{
    public static class MLHelper
    {
        public static TrainTestData LoadDataFromTextFile<T>(MLContext mlContext, string path, 
            bool hasHeader = false, double testFraction = 0.1)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<T>(path, hasHeader: hasHeader);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: testFraction);
            return splitDataView;
        }        
    }
}
