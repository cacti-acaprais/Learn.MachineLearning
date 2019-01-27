using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Learn.MachineLearning.Tests
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestMethod1()
        {
            var mlContext = new MLContext();
            string dataPath = "iris-data.txt";

            var reader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column(nameof(IrisData.SepalLength), DataKind.R4, 0),
                    new TextLoader.Column(nameof(IrisData.SepalWidth), DataKind.R4, 1),
                    new TextLoader.Column(nameof(IrisData.PetalLength), DataKind.R4, 2),
                    new TextLoader.Column(nameof(IrisData.PetalWidth), DataKind.R4, 3),
                    new TextLoader.Column(nameof(IrisData.Label), DataKind.Text, 4)
                }
            });

            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(IrisData.Label))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth), nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingDataView);

            var predictionEngine = model.MakePredictionFunction<IrisData, IrisPrediction>(mlContext);

            var prediction = predictionEngine.Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Assert.IsFalse(string.IsNullOrEmpty(prediction.PredictionLabels));
        }
    }
}
