
using BNN;
using BNN.Tests;

// SingleNeuron.Run();
// XorTest.Run();

var rand = new Random();
var nums = rand.Randn(50);
foreach (var num in nums)
{
    Console.Write(num+", ");
}

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

string DisplayArray(double[] arr)
{
    return "[" + string.Join(",", arr) + "]";
}


double[][] BuildTrainingSet(int numberOfSamples)
{
    double[][] samples = new double[numberOfSamples][];

    var rand = new Random();
    
    for (int i = 0; i < numberOfSamples; i++)
    {
        double a = rand.NextDouble();
        double b = rand.NextDouble();
        double y = Math.Pow(a, 2) + b;
        double[] sample = new[] {a, b, y};
        samples[i] = sample;
    }

    return samples;
}

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

/*
 * https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
 * https://cs231n.github.io/neural-networks-case-study/
 */
double[,] BuildSpiralData(int numberOfSamples, int numberOfClasses)
{
    /*
     * for any given sample n
     * result[n,0] = X coord
     * result[n,1] = Y coord
     * result[n,2] = the class (ie dataset) the point belongs to 
     */
    var results = new double[numberOfSamples * numberOfClasses,3];

    foreach (var class_number in Enumerable.Range(0,numberOfClasses))
    {
        for (var n = numberOfSamples * class_number; n < numberOfSamples * (class_number + 1); n++)
        {
            
        }
    }
    
    return results;
}

// return evenly spaced numbers between start and end
double[] Linespace(double start, double end, int num)
{
    var delta = (end - start) / num;
    var results = new double[num];
    var value = start;
    for (var i = 0; i < num; i++)
    {
        results[i] = value;
        value += delta;
    }

    return results;
}