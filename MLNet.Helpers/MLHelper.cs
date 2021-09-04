using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNet.Helpers
{
    public static class MLHelper
    {
        public static TrainTestData LoadDataFromTextFile<T>(MLContext mlContext, string path,
            double testFraction, bool hasHeader = false, char separatorChar = '\t')
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<T>(path, hasHeader: hasHeader, separatorChar: separatorChar);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: testFraction);
            return splitDataView;
        }

        public static IDataView LoadDataFromTextFile<T>(MLContext mlContext, string path,
            bool hasHeader = false, char separatorChar = '\t')
        {
            return mlContext.Data.LoadFromTextFile<T>(path, hasHeader: hasHeader, separatorChar: separatorChar);
        }
    }
}
