namespace BNN.Tests;

public class SingleNeuron
{
    public static void Run()
    {
        var toArray = (double x) => new[] {x};

        var network = NetworkBuilder
            .WithInputs(3)
            .WithLayer(1, ActivationFunctions.Tanh)
            .Build();
        var trainingInput = new [] {new[] {0.0, 0.0, 1.0}, new[] {1.0, 1.0, 1.0}, new[] {1.0, 0.0, 1.0}, new[] {0.0, 1.0, 1.0}};
        var trainingExpected = new[] {0.0, 1.0, 1.0, 1.0};
        var testInput = new[] {1.0, 0.0, 0.0};
        var testExpected = 1.0;

        Console.WriteLine(network.Dump());
        var avgErr=0.0;
        for (int e = 0; e < 5001; e++)
        {
            for(int s=0; s<trainingInput.Length;s++)
            {
                var sample = trainingInput[s];
                avgErr = network.Train(sample, toArray(trainingExpected[s]), 0.1);
            }
            if (e%100==0) Console.WriteLine($"average error for epoch {e}: {avgErr}");
        }
        Console.WriteLine($"final average error: {avgErr}");

        Console.WriteLine(network.Dump());

        var predicted = network.Apply(testInput);
        Console.WriteLine($"Test prediction: {predicted[0]}");
    }
}