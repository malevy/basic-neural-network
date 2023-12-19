namespace BNN.Tests;

using Plotly.NET;

public class VerticalTest
{
    public static void Run()
    {
        var sampleCount = 200;
        var rand = new Random();
        var trainingInputs = DataGenerators.BuildVerticalDataSet(sampleCount, 3);

        var network = NetworkBuilder.WithInputs(2)
            .WithLayer(2, new ActivationFunctions.LeakyReLuFunction(0.02), 0.9)
            .WithLayer(3, new ActivationFunctions.SoftmaxFunction())
            .WithGradientLossFunction(LossFunctions.CategoricalCrossEntropyDerivative)
            .WithAggregateLossFunction(LossFunctions.CategoricalCrossEntropy)
            .Build();

        var learningRate = new LearningRate(0.06, 5e-5);
        List<double> errors = new();

        // train
        var err = 0.0;
        for (var e = 0; e < 10001; e++)
        {
            // Shuffle(trainingInputs);

            for (var n = 0; n < trainingInputs.GetLength(0); n++)
            {
                var expected = new[] {0.0, 0.0, 0.0};
                expected[(int) trainingInputs[n, 2]] = 1.0;
                var inputs = new[]
                {
                    trainingInputs[n, 0],
                    trainingInputs[n, 1]
                };
                err = network.Train(inputs, expected, learningRate.Value);
            }

            if (e % 100 == 0)
            {
                Console.WriteLine($"epoch:{e} error:{err} lr:{learningRate.Value}");
                errors.Add(err);
            }

            learningRate.Decay();
        }

        Console.WriteLine($"last error:{err}");
        errors.Add(err);
        ErrorGraph.Graph(errors);

        // test
        var testSamples = 50;
        var testingInputs = DataGenerators.BuildVerticalDataSet(testSamples, 3);
        var correct = 0.0;
        Dictionary<int, List<Tuple<double, double>>> predictedLists = new()
        {
            {0, new List<Tuple<double, double>>()},
            {1, new List<Tuple<double, double>>()},
            {2, new List<Tuple<double, double>>()}
        };
        Shuffle(testingInputs);
        for (var i = 0; i < testingInputs.GetLength(0); i++)
        {
            var inputs = new[]
            {
                testingInputs[i, 0],
                testingInputs[i, 1]
            };

            var predicted = network.Apply(inputs);
            if (Math.Abs(ArrayUtils.ArgMax(predicted) - testingInputs[i, 2]) < 0.0001) correct++;

            predictedLists[ArrayUtils.ArgMax(predicted)]
                .Add(new Tuple<double, double>(testingInputs[i, 0], testingInputs[i, 1]));
        }

        Console.WriteLine($"Accuracy: {correct / testingInputs.GetLength(0)}");


        // plot the results
        Dictionary<int, List<Tuple<double, double>>> actualLists = new()
        {
            {0, new List<Tuple<double, double>>()},
            {1, new List<Tuple<double, double>>()},
            {2, new List<Tuple<double, double>>()}
        };

        for (int i = 0; i < testingInputs.GetLength(0); i++)
        {
            var point = new Tuple<double, double>(testingInputs[i, 0], testingInputs[i, 1]);
            actualLists[(int) testingInputs[i, 2]].Add(point);
        }

        var charts = new[]
        {
            Chart2D.Chart.Scatter<double, double, string>(Name: "actual class 0",
                xy: actualLists[0], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red"),
                MarkerSymbol: StyleParam.MarkerSymbol.Diamond, Opacity: 0.3
            ),
            Chart2D.Chart.Scatter<double, double, string>(Name: "actual class 1",
                xy: actualLists[1], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("green"),
                MarkerSymbol: StyleParam.MarkerSymbol.Diamond, Opacity: 0.3
            ),
            Chart2D.Chart.Scatter<double, double, string>(Name: "actual class 2",
                xy: actualLists[2], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("blue"),
                MarkerSymbol: StyleParam.MarkerSymbol.Diamond, Opacity: 0.3
            ),
            Chart2D.Chart.Scatter<double, double, string>(Name: "predicted class 0",
                xy: predictedLists[0], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red"),
                MarkerSymbol: StyleParam.MarkerSymbol.CircleX
            ),
            Chart2D.Chart.Scatter<double, double, string>(Name: "predicted class 1",
                xy: predictedLists[1], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("green"),
                MarkerSymbol: StyleParam.MarkerSymbol.CircleX
            ),
            Chart2D.Chart.Scatter<double, double, string>(Name: "predicted class 2",
                xy: predictedLists[2], mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("blue"),
                MarkerSymbol: StyleParam.MarkerSymbol.CircleX
            )
        };
        Chart.Combine(charts).Show();
    }


    static void Shuffle(double[,] data)
    {
        var rand = new Random();
        var n = data.GetLength(0) - 1;
        while (n > 1)
        {
            var s = rand.Next(n);

            var d = (data[n, 0], data[n, 1], data[n, 2]);
            (data[n, 0], data[n, 1], data[n, 2]) = (data[s, 0], data[s, 1], data[s, 2]);
            (data[s, 0], data[s, 1], data[s, 2]) = d;

            n--;
        }
    }
}