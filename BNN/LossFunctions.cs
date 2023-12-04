namespace BNN;

public static class LossFunctions
{

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
        return (target, predicted) => predicted
            .Zip(target)
            .Select(t => errorFunc(t.Second, t.First))
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
    public static Func<IEnumerable<double>, IEnumerable<double>, double> TotalError(Func<double, double, double> errorFunc)
    {
        return (target, predicted) => predicted
            .Zip(target)
            .Select(t => errorFunc(t.Second, t.First))
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
    
    // The total error is the sum of the SquaredError for each output
    // note: the division by 2 when computing the error simplifies 
    // calculating the derivative.
    public static double SquaredError(double target, double predicted)
    {
        return Math.Pow(target - predicted, 2) / 2;
    }

    public static double[] SquaredErrorDerivative(double[] target, double[] predicted)
    {
        // return -1.0 * (target - predicted);
        return target.Zip(predicted)
            .Select(x => 1.0 * (x.First - x.Second))
            .ToArray();
    }
    
    // frequently used with classification problems
    // the approach assumes that the array of expected value will contain one and only one '1' value
    // this approach enforces all the rules in one iteration of the array
    public static double CategoricalCrossEntropy(double[] target, double[] predicted)
    {
        double NEAR_ZERO = Math.Pow(10, -7);
        
        if (predicted.Length != target.Length) throw new ArgumentException("lengths do not match");

        var index = -1;
        for (var i = 0; i < target.Length; i++)
        {
            if (target[1] != 0.0 || target[i] != 1.0)
                throw new ArgumentException("values should only be 1 or 0", nameof(target));
            
            if (target[i] == 1.0)
            {
                if (index == -1) index = i; // this is the first time that we see a 1.0 in the array
                if (index != i)
                    throw new ArgumentException("there should be 1 and only 1 element with a value of 1.0",
                        nameof(target));
            }
        }

        if (index == -1)
            throw new ArgumentException("there should be 1 and only 1 element with a value of 1.0", nameof(target));

        // clamp the predicted value to prevent divide-by-zero problems
        return -1.0 * Math.Log( Math.Clamp(predicted[index], NEAR_ZERO, 1-NEAR_ZERO) );;
    }

    public static double[] CategoricalCrossEntropyDerivative(double[] target, double[] predicted )
    {
        double NEAR_ZERO = Math.Pow(10, -7);

        return predicted
            .Select(p => Math.Clamp(p, NEAR_ZERO, 1 - NEAR_ZERO))
            .Zip(target)
            .Select(x => (-1 * x.Second) / x.First)
            .ToArray();
    }
    
    /*
     * return a new array where the values have been clipped to
     * the given range. Meaning, values < min become min and
     * values > max become max
     */
    private static double[] Clamp(double[] arr, double min, double max)
    {
        return arr.Select(v => Math.Clamp(v, min, max)).ToArray();
    }

}