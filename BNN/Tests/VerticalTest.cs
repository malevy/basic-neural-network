namespace BNN.Tests;

public class VerticalTest
{
    public static void Run()
    {
        var rand = new Random();
        var trainingInputs = DataGenerators.BuildVerticalDataSet(100, 3);

        var network = NetworkBuilder.WithInputs(2)
            .WithLayer(8, new ActivationFunctions.SigmoidFunction())
            .WithLayer(3, new ActivationFunctions.SoftmaxFunction())
            .WithGradientLossFunction(LossFunctions.CategoricalCrossEntropyDerivative)
            .WithAggregateLossFunction(LossFunctions.CategoricalCrossEntropy)
            .Build();

        double learningRate = 1.0;
        const double learningRateDecay = 0.0001;
        var iteration = 0;
        
        // train
        var err = 0.0;
        for (var e = 0; e < 10000; e++)
        {
            Shuffle(trainingInputs);

            double applicableLearningRate=0;
            for (var n = 0; n < trainingInputs.GetLength(0); n++)
            {
                // decay the learning rate a little bit with each iteration
                applicableLearningRate = learningRate * (1.0 / (1.0 + (learningRateDecay * iteration)));

                var expected = new[] { 0.0, 0.0, 0.0 };
                expected[(int)trainingInputs[n, 2]] = 1.0;
                var inputs = new[]
                {
                    trainingInputs[n, 0],
                    trainingInputs[n, 1]
                };
                err = network.Train(inputs, expected, applicableLearningRate);
                iteration++;
            }
            if (e % 100 == 0) Console.WriteLine($"epoch:{e} error:{err} lr:{applicableLearningRate}");

            if (err < 0.0001)
            {
                Console.WriteLine($"training stopped after {e}; error={err}");
                break;
            }
        }
        Console.WriteLine($"last error:{err}");

        // test
        for (var i = 0; i < 10; i++)
        {
            var s = rand.Next(50);
            var inputs = new[]
            {
                trainingInputs[s, 0],
                trainingInputs[s, 1]
            };
            var expected = new[] { 0.0, 0.0, 0.0 };
            expected[(int)trainingInputs[s, 2]] = 1.0;

            var predicted = network.Apply(inputs);
            Console.WriteLine($"expected {DisplayArray(expected)}, predicted {DisplayArray(predicted)}");
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