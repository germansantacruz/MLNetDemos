﻿using Microsoft.ML.Data;

namespace MLNet.D005.LogisticRegression.ML
{
    public class FileInput
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string Strings { get; set; }
    }
}
