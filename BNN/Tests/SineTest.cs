﻿using Plotly.NET;

namespace BNN.Tests;

public class SineTest
{
    public static void Run()
    {
        var network = NetworkBuilder.WithInputs(1)
            .WithLayer(25, new ActivationFunctions.LeakyReLuFunction(0.01), 0.5)
            .WithLayer(25, new ActivationFunctions.LeakyReLuFunction(0.01), 0.5)
            .WithLayer(1, new ActivationFunctions.LinearFunction())
            .WithGradientLossFunction(LossFunctions.SquaredErrorDerivative)
            .WithAggregateLossFunction(LossFunctions.MeanError(LossFunctions.SquaredError))
            .Build();

        var learningRate = new LearningRate(0.05, 0.001);
        List<double> errors = new();

        // train
        var sampleCount = 100;
        var data = DataGenerators.BuildSineData(sampleCount);
        var err = 0.0;
        for (var e = 0; e < 10001; e++)
        {
            // ArrayUtils.Shuffle(data);

            for (var n = 0; n < data.GetLength(0); n++)
            {
                var expected = new[] { data[n, 1] }; // sin(x)
                var inputs = new[] { data[n, 0] }; // X
                err = network.Train(inputs, expected, learningRate.Value);
                if (Double.IsNaN(err)) throw new ApplicationException("training failed. error is NaN");
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
        var correct = 0.0;
        sampleCount = 100;
        var testingInputs = DataGenerators.BuildSineData(sampleCount);
        List<Tuple<double, double>> predictedList = new();
        // ArrayUtils.Shuffle(testingInputs);
        for (var i = 0; i < sampleCount; i++)
        {
            var inputs = new[] { testingInputs[i, 0], };
            var predicted = network.Apply(inputs);
            if (Math.Abs(predicted[0] - testingInputs[i, 1]) < 0.0001) correct++;
            Console.WriteLine($"predicted: {predicted[0]} actual: {testingInputs[i, 1]}");
            predictedList.Add(new Tuple<double, double>(testingInputs[i, 0], predicted[0]));
        }

        Console.WriteLine($"Accuracy: {correct / sampleCount}");
        
        // graph test vs predicted
        var series0 = Enumerable.Range(0, data.GetLength(0))
         .Select(index => new Tuple<double,double>(testingInputs[index, 0], testingInputs[index, 1]))
         .ToArray();
        
        var charts = new[]
        {
         Chart2D.Chart.Scatter<double, double, string>(
          xy: series0, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("black")
         ),
         Chart2D.Chart.Scatter<double, double, string>(
             xy: predictedList, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red")
         ),
        };
        Chart.Combine(charts).Show();

    }
}