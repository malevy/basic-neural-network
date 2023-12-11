
using BNN;
using Plotly.NET;

// SingleNeuron.Run();
 // XorTest.Run();
//VerticalTest.Run();

// var x = new[] { 1.0, 2.0, 3.0 };
// var y = new[] { 4.0, 5.0, 6.0 };
// var z = String.Join(',', x.Zip(y)
//     .Select(a => a.First * a.Second));
// Console.WriteLine(z);

// var network = NetworkBuilder
//     .WithInputs(2)
//     .WithLayer(16, ActivationFunctions.ReLU)
//     .WithLayer(16, ActivationFunctions.ReLU)
//     .WithOutput(1)
//     .Build();
// solving for y = a^2+b
// ex. f(a,b) = f(4,3) = 19;
// var testInputs = new double[]{4, 3};
// var testTargets = new double[19];
// double[][] samples = BuildTrainingSet(100);
// var trainingInput = samples.Select(s => new double[] {s[0], s[1]}).ToArray();
// var trainingExpected = samples.Select(s => new double[] {s[2]}).ToArray();


// learn sin(x)
// var sinData = BuildSineData(100);
//
// var network = NetworkBuilder
//     .WithInputs(1)
//     .WithLayer(64, ActivationFunctions.ReLU)
//     .WithLayer(64, ActivationFunctions.ReLU)
//     .WithOutput(1)
//     .Build();

// Console.WriteLine(network.Dump());
//
// var avgErr=0.0;
// for (int e = 0; e < 5001; e++)
// {
//     for(int s=0; s<sinData.Length;s++)
//     {
//         var sample = sinData[s];
//         avgErr = network.Train(toArray(sample[0]), toArray(sample[1]), 0.00001);
//     }
//     if (e%100==0) Console.WriteLine($"average error for epoch {e}: {avgErr}");
// }
// Console.WriteLine($"final average error: {avgErr}");
//
// Console.WriteLine(network.Dump());
//
// var totalError = 0.0;
// foreach (var testData in sinData)
// {
//     var predicted = network.Apply(toArray(testData[0]));
//     totalError = LossFunctions.AbsoluteError(testData[1], predicted[0]);
//     Console.WriteLine($"{testData[0]},{testData[1]},{predicted[0]}");
// }
// Console.WriteLine($"Avg error during testing: {totalError/sinData.Length}");

double[][] BuildSineData(int numberOfSamples)
{
    double[][] samples = new double[numberOfSamples][];

    var rand = new Random();
    for (int i = 0; i < numberOfSamples; i++)
    {
        var x = rand.NextDouble(-1, 1);
        var y = Math.Sin(x);
        var sample = new[] {x, y};
        samples[i] = sample;
    }

    return samples;
    
}

