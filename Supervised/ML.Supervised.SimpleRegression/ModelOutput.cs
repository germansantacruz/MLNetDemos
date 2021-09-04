using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.Supervised.SimpleRegression
{
    public class ModelOutput 
    {   
        [ColumnName("Score")]
        public float Salary;
    }
}
