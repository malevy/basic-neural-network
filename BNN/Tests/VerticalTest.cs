namespace BNN.Tests;

public class VerticalTest
{
    public static void Run()
    {
        var sampleCount = 100;
        var rand = new Random();
        var trainingInputs = DataGenerators.BuildVerticalDataSet( sampleCount, 3);

        var network = NetworkBuilder.WithInputs(2)
            .WithLayer(100, new ActivationFunctions.LeakyReLuFunction(0.01), 0.5)
            .WithLayer(3, new ActivationFunctions.SoftmaxFunction())
            .WithGradientLossFunction(LossFunctions.CategoricalCrossEntropyDerivative)
            .WithAggregateLossFunction(LossFunctions.CategoricalCrossEntropy)
            .Build();

        var learningRate = new LearningRate(3e-4, 0.00001);
        List<double> errors = new();
        
        // train
        var err = 0.0;
        for (var e = 0; e < 10001; e++)
        {
            Shuffle(trainingInputs);

            for (var n = 0; n < trainingInputs.GetLength(0); n++)
            {
                var expected = new[] { 0.0, 0.0, 0.0 };
                expected[(int)trainingInputs[n, 2]] = 1.0;
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
            if (err < 0.001)
            {
                Console.WriteLine($"training stopped after {e}; error={err}");
                break;
            }
            learningRate.Decay();

        }
        Console.WriteLine($"last error:{err}");
        errors.Add(err);
        ErrorGraph.Graph(errors);

        // test
        var testSamples = 10;
        var testingInputs = DataGenerators.BuildVerticalDataSet( testSamples, 3);

        Shuffle(testingInputs);
        for (var i = 0; i < testSamples; i++)
        {
            var s = rand.Next(10);
            var inputs = new[]
            {
                testingInputs[s, 0],
                testingInputs[s, 1]
            };
            var expected = new[] { 0.0, 0.0, 0.0 };
            expected[(int)testingInputs[s, 2]] = 1.0;

            var predicted = network.Apply(inputs);
            var loss = LossFunctions.CategoricalCrossEntropy(expected, predicted);
            Console.WriteLine($"test: {DisplayArray(inputs)}\t predicted {DisplayArray(predicted)}\t expected {DisplayArray(expected)}\t loss:{loss}");
        }
        
        
    }

    static string DisplayArray(double[] arr)
    {
        return "[" + string.Join(",", arr) + "]";
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