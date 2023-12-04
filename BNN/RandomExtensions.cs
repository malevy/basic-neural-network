namespace BNN;

public static class RandomExtensions
{
    public static double NextDouble(this Random @this, double min, double max)
    {
        return (@this.NextDouble() * (max - min)) + min;
    }

    /**
     * return a random item from the given list
     */
    public static T Random<T>(this Random @this, T[] items)
    {
        return items[@this.Next(items.Length)];
    }

    /**
     * Return 'num' random number between 0 (inclusive) and
     * 1 (exclusive) with a normal distribution of mean=0
     * and variance=1.
     *
     * this method simulates Numpy's "random.randn" function
     * with one dimension
     */
    public static double[] Randn(this Random @this, int num)
    {
        var mean=0.0;
        var stdDev = 1.0;
        var results = new double[num];
        for (var i = 0; i < num; i++)
        {
            var r1 = @this.NextDouble();
            var r2 = @this.NextDouble();
            var ranStdNorm =
                Math.Sqrt(-2.0 * Math.Log(r1)) *
                Math.Sin(2.0 * Math.PI * r2);
            var randNorm = (stdDev * ranStdNorm) + mean;
            results[i] = randNorm;
        }

        return results;
    }
}