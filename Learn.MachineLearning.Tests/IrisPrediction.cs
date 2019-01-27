using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace Learn.MachineLearning.Tests
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictionLabels;
    }
}
