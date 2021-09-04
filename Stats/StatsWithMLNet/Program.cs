using Microsoft.Data.Analysis;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using XPlot.Plotly;
using FSharp.Stats;
using MathNet.Numerics.Statistics;

namespace StatsWithMLNet
{
    class Program
    {
        static void Main(string[] args)
        {
            DataFrame df = new DataFrame();

            var max = df * 2;

            List<float> data = new List<float>();
            data.Average();

            var ly = new Layout.Layout()
            {
                
            }


            df.GroupBy("").First("");

            DataFrame.LoadCsv(".csv", separator: ' ', header: false,
                dataTypes: new[] { Type.GetType("double"), Type.GetType("double"), Type.GetType("double"), Type.GetType("double"), Type.GetType("double"),
                                   Type.GetType("double"), Type.GetType("double"), Type.GetType("double"), Type.GetType("double"), Type.GetType("double")});

            Console.ReadLine();
        }       
    }
}
