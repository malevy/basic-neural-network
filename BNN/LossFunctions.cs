namespace BNN;

public static class LossFunctions
{

    // public record ErrorPair(double Predicted, double Target);

    // calculate the average of the errors based on the given error function
    public static double MeanError(IEnumerable<double> actual, IEnumerable<double> target, Func<double, double, double> errorFunc)
    {
        return actual
            .Zip(target)
            .Select(t => errorFunc(t.Second, t.First))
            .Average();
    }

    // calculate the sum of the errors based on the given error function
    public static double TotalError(IEnumerable<double> actual, IEnumerable<double> target, Func<double, double, double> errorFunc)
    {
        return actual
            .Zip(target)
            .Select(t => errorFunc(t.Second, t.First))
            .Sum();
    }
    
    // frequently used with regression problems
    // Mean Absolute Error is the average of all the errors
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

    public static double dSquaredError(double target, double predicted)
    {
        return -1.0 * (target - predicted);
    }
    
    // frequently used with classification problems
    public static double CategoricalCrossEntropy(double[] predicted, double[] expected)
    {
        if (predicted.Length != expected.Length) throw new ArgumentException("lengths do not match");

        // the approach assumes that the array of expected value will contain one and only one '1' value
        // this approach enforces all the rules in one iteration of the array
        var index = -1;
        for (var i = 0; i < expected.Length; i++)
        {
            if (expected[1] != 0.0 || expected[i] != 1.0)
                throw new ArgumentException("values should only be 1 or 0", nameof(expected));
            if (expected[i] == 1.0)
            {
                if (index == -1) index = i; // this is the first time that we see a 1.0 in the array
                if (index != i)
                    throw new ArgumentException("there should be 1 and only 1 element with a value of 1.0",
                        nameof(expected));
            }
        }

        if (index == -1)
            throw new ArgumentException("there should be 1 and only 1 element with a value of 1.0", nameof(expected));

        return -1.0 * Math.Log(predicted[index]);
    }
}