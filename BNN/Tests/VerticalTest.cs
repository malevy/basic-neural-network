namespace BNN.Tests;

public class VerticalTest
{
    public static void Run()
    {
        var rand = new Random();
        var trainingInputs = BuildVerticalDataSet(50, 3);

        var network = NetworkBuilder.WithInputs(2)
            .WithLayer(3, new ActivationFunctions.ReLuFunction())
            .WithLayer(3, new ActivationFunctions.SoftmaxFunction())
            .WithGradientLossFunction(LossFunctions.CategoricalCrossEntropyDerivative)
            .WithAggregateLossFunction(LossFunctions.CategoricalCrossEntropy)
            .Build();

        // train
        for (var e = 0; e < 4000; e++)
        {
            Shuffle(trainingInputs);

            var err = 0.0;
            for (var n = 0; n < trainingInputs.GetLength(0); n++)
            {
                var expected = new double[3];
                expected[(int)trainingInputs[n, 2]] = 1.0;
                err = network.Train(
                    new[]
                    {
                        trainingInputs[n, 0],
                        trainingInputs[n, 1]
                    },
                    expected, 0.25);
                if (e % 100 == 0) Console.WriteLine($"error = {err}");
            }

            if (err < 0.0001)
            {
                Console.WriteLine($"training stopped after {e}; error={err}");
                break;
            }
        }

        // test
        for (var i = 0; i < 10; i++)
        {
            var s = rand.Next(50);
            var inputs = new[]
            {
                trainingInputs[s, 0],
                trainingInputs[s, 1]
            };
            var expected = new double[3];
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

    /*
     * based on https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/vertical.py
     * for any given sample n
     * result[n,0] = X coord
     * result[n,1] = Y coord
     * result[n,2] = the class (ie dataset) the point belongs to
     */
    static double[,] BuildVerticalDataSet(int sampleCount, int classCount)
    {
        var rand = new Random();
        double[,] results = new double[sampleCount * classCount, 3];

        for (var c = 0; c < classCount; c++)
        {
            var offset = c * sampleCount;
            for (var n = 0; n < sampleCount; n++)
            {
                var X = rand.Randn(sampleCount).Select(v => (v * 0.1) + (c / 3.0)).ToArray();
                var Y = rand.Randn(sampleCount).Select(v => (v * 0.1) + 0.5).ToArray();
                results[offset + n, 0] = X[n];
                results[offset + n, 1] = Y[n];
                results[offset + n, 2] = c;
            }
        }

        return results;
    }
}