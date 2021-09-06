using System;
using System.IO;

namespace MLNet.D005.LogisticRegression.ML
{
    public class FeatureExtractor : BaseML
    {
        public void Extract(string folderPath)
        {
            var files = Directory.GetFiles(folderPath);

            using (var streamWriter = new StreamWriter(dataPath))
            {
                foreach (var file in files)
                {
                    var strings = GetStrings(File.ReadAllBytes(file));
                    streamWriter.WriteLine($"{file.ToLower().Contains("malicious")}\t{strings}");
                }
            }

            Console.WriteLine($"Extracted {files.Length} to {dataPath}");
        }
    }
}
