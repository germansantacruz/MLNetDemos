using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace MLNet.D005.LogisticRegression.ML
{
    public class BaseML
    {
        static readonly string DATA_FILENAME = "sampledata.csv";
        static readonly string MODEL_FILENAME = "demo005.zip";

        protected readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", DATA_FILENAME);
        protected readonly string modelPath = Path.Combine(Environment.CurrentDirectory, "Models", MODEL_FILENAME);

        protected readonly MLContext mlContext;
        private readonly Regex _stringRex;

        protected BaseML()
        {
            mlContext = new MLContext(2020);

            // Para utilizar la codificación Windows-1252
            // Esta codificación es la que utilizan los ejecutables de Windows
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);            
            _stringRex = new Regex(@"[ -~\t]{8,}", RegexOptions.Compiled);
        }

        protected string GetStrings(byte[] data)
        {
            var stringLines = new StringBuilder();

            if (data == null || data.Length == 0)
            {
                return stringLines.ToString();
            }

            using (var ms = new MemoryStream(data, false))
            {
                using (var streamReader = new StreamReader(ms, Encoding.GetEncoding(1252), false, 2048, false))
                {
                    while (!streamReader.EndOfStream)
                    {
                        var line = streamReader.ReadLine();

                        if (string.IsNullOrEmpty(line))
                        {
                            continue;
                        }

                        line = line.Replace("^", "").Replace(")", "").Replace("-", "");
                        stringLines.Append(string.Join(string.Empty,
                            _stringRex.Matches(line).Where(a => !string.IsNullOrEmpty(a.Value) && !string.IsNullOrWhiteSpace(a.Value)).ToList()));
                    }
                }
            }

            return string.Join(string.Empty, stringLines);
        }
    }
}
