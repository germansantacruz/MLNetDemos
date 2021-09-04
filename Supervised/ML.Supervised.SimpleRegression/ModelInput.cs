using Microsoft.ML.Data;

namespace ML.Supervised.SimpleRegression
{
    public class ModelInput
    {
        [LoadColumn(0), ColumnName("YearsOfExperience")]
        public float YearsOfExperience;

        [LoadColumn(1), ColumnName("Salary")]
        public float Salary;
    }
}
