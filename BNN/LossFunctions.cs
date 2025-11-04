namespace BNN;

public static class LossFunctions
{
    public static double NEAR_ZERO = Math.Pow(10.0, -10);

    /// <summary>
    /// calculate the average of the errors based on the given error function
    /// </summary>
    /// <param name="errorFunc">
    /// The function used to calculate the error. The function takes two doubles, target and predicted
    /// and returns the error
    /// </param>
    /// <returns>the error</returns>
    public static Func<double[], double[], double> MeanError(Func<double, double, double> errorFunc)
    {
        return (target, predicted) => target
            .Zip(predicted)
            .Select(t => errorFunc(t.First, t.Second))
            .Average();
    }

    /// <summary>
    /// calculate the sum of the errors based on the given error function
    /// </summary>
    /// <param name="errorFunc">
    /// The function used to calculate the error. The function takes two doubles, target and predicted
    /// and returns the error
    /// </param>
    /// <returns>the error</returns>
    public static Func<double[], double[], double> TotalError(Func<double, double, double> errorFunc)
    {
        return (target, predicted) => target
            .Zip(predicted)
            .Select(t => errorFunc(t.First, t.Second))
            .Sum();
    }

    // frequently used with regression problems
    public static double AbsoluteError(double target, double predicted)
    {
        return Math.Abs(target - predicted);
    }

    public static double DiffError(double target, double predicted)
    {
        return target - predicted;
    }

    public static double SquaredError(double target, double predicted)
    {
        return Math.Pow(target - predicted, 2);
    }

    public static double[] SquaredErrorDerivative(double[] target, double[] predicted)
    {
        // return -1.0 * (target - predicted);
        return target
            .Zip(predicted)
            .Select(x => -2.0 * (x.First - x.Second))
            .ToArray();
    }

    // assume we're working on a multi-label classification problem
    // and the target is a one-hot vector, such that a sample belongs to one
    // and only one class.
    public static double CategoricalCrossEntropy(double[] target, double[] predicted)
    {
        if (predicted.Length != target.Length) throw new ArgumentException("lengths do not match");

        var loss = -1.0 * target
            .Zip(predicted)
            .Select(x => Math.Log(x.Second + NEAR_ZERO) * x.First)
            .Sum();

        return loss;
    }

    public static double[] CategoricalCrossEntropyDerivative(double[] target, double[] predicted)
    {
        return predicted
            .Zip(target)
            .Select(x => (-1.0 * x.Second) / (x.First + NEAR_ZERO))
            .ToArray();
    }

    public static double BinaryCrossEntropy(double[] target, double[] predicted)
    {
        if (predicted.Length != target.Length) throw new ArgumentException("lengths do not match");

        return predicted
            .Select(p => Math.Clamp(p, NEAR_ZERO, 1.0 - NEAR_ZERO))
            .Zip(target)
            .Select(x => -1.0 * (x.Second * Math.Log(x.First) + (1.0 - x.Second) * Math.Log(1.0 - x.First)))
            .Average();
    }

    public static double[] BinaryCrossEntropyDerivative(double[] target, double[] predicted)
    {
        var numberOfOutputs = predicted.Length;
        return predicted
            .Select(p => Math.Clamp(p, NEAR_ZERO, 1 - NEAR_ZERO))
            .Zip(target)
            .Select(x => -1.0 * (x.Second / x.First - (1 - x.Second) / (1 - x.First)) / numberOfOutputs)
            .ToArray();
    }
}