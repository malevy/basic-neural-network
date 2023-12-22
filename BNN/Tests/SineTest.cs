using Plotly.NET;

namespace BNN.Tests;

public class SineTest
{
    public static void Run()
    {
        // results are better when the same number of neurons is used for both layers
        var network = NetworkBuilder.WithInputs(1)
            .WithLayer(100, new ActivationFunctions.TanhFunction(),0.6)
            .WithLayer(1, new ActivationFunctions.LinearFunction())
            .WithGradientLossFunction(LossFunctions.SquaredErrorDerivative)
            .WithAggregateLossFunction(LossFunctions.MeanError(LossFunctions.SquaredError))
            .Build();

        var learningRate = new LearningRate(3e-3, 0.0001);
        List<double> errors = new();
        var sampleCount = 600;

        double[] expected = { 0.0 };
        double[] inputs = { 0.0 };
        
        // train
        var err = 0.0;
        for (var e = 0; e < 10001; e++) //10000
        {
            var data = DataGenerators.BuildSineData(sampleCount);

            for (var n = 0; n < data.GetLength(0); n++)
            {
                expected[0] = data[n, 1]; // sin(x)
                inputs[0] = data[n, 0]; // X
                err = network.Train(inputs, expected, learningRate.Value);
                if (Double.IsNaN(err))
                {
                    Console.WriteLine(network.Dump());
                    throw new ApplicationException($"training failed. error is NaN epoch: {e} sample: {n}");
                }
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
            inputs[0] = testingInputs[i, 0];
            var predicted = network.Apply(inputs);
            if (Math.Abs(predicted[0] - testingInputs[i, 1]) < 0.01) correct++;
            Console.WriteLine($"predicted: {predicted[0]} actual: {testingInputs[i, 1]}");
            predictedList.Add(new Tuple<double, double>(testingInputs[i, 0], predicted[0]));
        }

        Console.WriteLine($"Accuracy: {(correct / sampleCount):F5} {correct} out of {sampleCount} ");
        
        // graph test vs predicted
        var series0 = Enumerable.Range(0, testingInputs.GetLength(0))
         .Select(index => new Tuple<double,double>(testingInputs[index, 0], testingInputs[index, 1]))
         .ToArray();
        
        var charts = new[]
        {
         Chart2D.Chart.Scatter<double, double, string>(Name: "actual",
          xy: series0, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("black")
         ),
         Chart2D.Chart.Scatter<double, double, string>(Name: "predicted",
             xy: predictedList, mode: StyleParam.Mode.Lines_Markers, MarkerColor: Color.fromString("red"), Opacity:0.50  
         ),
        };
        Chart.Combine(charts).Show();

    }
}