namespace BNN.Tests;

public class XorTest
{
    public static void Run()
    {
        var trainInputs = new double[][]
        {
            // a, b, a^b
            new[] { 0.0, 0.0, 0.0 },
            new[] { 0.0, 1.0, 1.0 },
            new[] { 1.0, 0.0, 1.0 },
            new[] { 1.0, 1.0, 0.0 },
        };

        var network = NetworkBuilder
            .WithInputs(2)
            .WithAggregateLossFunction(LossFunctions.MeanError(LossFunctions.SquaredError))
            .WithGradientLossFunction(LossFunctions.SquaredErrorDerivative)
            .WithLayer(2, new ActivationFunctions.ReLuFunction())
            .WithLayer(1, new ActivationFunctions.SigmoidFunction())
            .Build();

        Console.WriteLine(network.Dump());

        var rand = new Random();
        bool stopTraining = false;

        // train
        for (int e = 0; e < 4000 && !stopTraining; e++)
        {
            // var sample = rand.Random(trainInputs);

            double err=0.0;
            foreach (var sample in trainInputs)
            {
                err = network.Train(
                    new[] { sample[0], sample[1] },
                    new[] { sample[2] }, 0.15);
                if (e % 100 == 0) Console.WriteLine($"error = {err}");
                if (err < 0.0001)
                {
                    Console.WriteLine($"training stopped after {e}; error={err}");
                    stopTraining = true;
                    break;
                }

            }

        }

        // test
        foreach (var input in trainInputs)
        {
            var predicted = network.Apply(new[] { input[0], input[1] });
            Console.WriteLine($"{input[0]} ^ {input[1]} = {predicted[0]}, expected: {input[2]}");
        }
    }
}