namespace BNN.Tests;

public class SpiralTest
{
    /*
     * https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
     * https://cs231n.github.io/neural-networks-case-study/
     * for any given sample n
     * result[n,0] = X coord
     * result[n,1] = Y coord
     * result[n,2] = the class (ie dataset) the point belongs to
     */
    public static double[,] BuildSpiralData(int numberOfSamples, int numberOfClasses)
    {
        var results = new double[numberOfSamples * numberOfClasses, 3];

        for (var classNumber=0; classNumber<numberOfClasses; classNumber++)
        {
            var radius = Linespace(0.0, 1.0, numberOfSamples);
            var thetas = CalculateThetas(classNumber, numberOfSamples);
            var offset = classNumber * numberOfSamples;
            for (int ix = 0; ix < numberOfSamples; ix++)
            {
                results[ix + offset,0] = radius[ix] * Math.Sin(thetas[ix] * 2.5); // X coord
                results[ix + offset,1] = radius[ix] * Math.Cos(thetas[ix] * 2.5); // Y coord
                results[ix + offset,2] = classNumber;
            }
        }

        return results;

        double[] CalculateThetas(int classNumber, int numOfSamples)
        {
            var rand = new Random();
            var offsets = Linespace(classNumber * 4, (classNumber + 1) * 4, numberOfSamples);
            var deltas = rand.Randn(numOfSamples).Select(d => d * 0.2);
            var thetas = offsets.Zip(deltas).Select(p => p.First + p.Second);
            return thetas.ToArray();
        }
    }

// return evenly spaced numbers between start and end
    static double[] Linespace(double start, double end, int num)
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
}