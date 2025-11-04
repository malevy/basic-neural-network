namespace BNN.Tests;

public class SingleNeuron
{
    public static void Run()
    {
        var toArray = (double x) => new[] {x};

        var network = NetworkBuilder
            .WithInputs(3)
            .WithAggregateLossFunction(LossFunctions.MeanError(LossFunctions.AbsoluteError))
            .WithGradientLossFunction(LossFunctions.SquaredErrorDerivative)
            .WithLayer(1, new ActivationFunctions.TanhFunction())
            .Build();

        var trainingInput = new [] {new[] {0.0, 0.0, 1.0}, new[] {1.0, 1.0, 1.0}, new[] {1.0, 0.0, 1.0}, new[] {0.0, 1.0, 1.0}};
        var trainingExpected = new[] {0.0, 1.0, 1.0, 1.0};
        var testInput = new[] {1.0, 0.0, 0.0};

        Console.WriteLine(network.Dump());
        var averageError=0.0;
        for (int e = 0; e < 5001; e++)
        {
            for(int s=0; s<trainingInput.Length;s++)
            {
                var sample = trainingInput[s];
                averageError = network.Train(sample, 
                    toArray(trainingExpected[s]), 
                    0.1);
            }
            if (e%100==0) Console.WriteLine($"average error for epoch {e}: {averageError}");
        }
        Console.WriteLine($"final average error: {averageError}");

        Console.WriteLine(network.Dump());

        var predicted = network.Apply(testInput);
        Console.WriteLine($"Test prediction: {predicted[0]}");
    }
}