namespace BNN.Tests;

public class XorTest
{
    public static void Run()
    {
        var trainInputs = new double[][]
        {
            // a, b, a^b
            new[] {0.0, 0.0, 0.0},
            new[] {0.0, 1.0, 1.0},
            new[] {1.0, 0.0, 1.0},
            new[] {1.0, 1.0, 0.0},

        };

        var network = NetworkBuilder
            .WithInputs(2)
            .WithLayer(2, ActivationFunctions.ReLU)
            .WithLayer(1, ActivationFunctions.Sigmoid)
            .Build();

        var rand = new Random();
        
        // train
        for (int e = 0; e < 50000; e++)
        {
            var sample = rand.Random(trainInputs);
            var err = network.Train(new[] {sample[0], sample[1]}, new[] {sample[2]}, 0.15);
            if (e%100 == 0) Console.WriteLine($"error = {err}");
        }
        
        // test
        foreach (var input in trainInputs)
        {
            var predicted = network.Apply(new[] {input[0], input[1]});
            Console.WriteLine($"{input[0]} ^ {input[1]} = {predicted[0]}, expected: {input[2]}");
        }
    }
}