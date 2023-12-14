namespace BNN;

public static class ArrayUtils
{
    // return the index of the max value in the array
    public static int ArgMax(double[] arr)
    {
        return Array.IndexOf(arr, arr.Max());
    }

    public static void Shuffle(double[,] data)
    {
        var rand = new Random();
        var n = data.GetLength(0) - 1;
        while (n > 1)
        {
            Swap(data, rand.Next(n), n);
            n--;
        }
    }

    public static void Swap(double[] data, int source, int target)
    {
        (data[source], data[target]) = (data[target], data[source]);
    }

    public static void Swap(double[,] data, int source, int target)
    {
        if (data.Rank > 3) throw new Exception("Data must be 2D or 3D array");
        if (data.Rank == 3) (data[source, 2], data[target, 2]) = (data[target, 2], data[source, 2]);
        if (data.Rank >= 2) (data[source, 1], data[target, 1]) = (data[target, 1], data[source, 1]);
        if (data.Rank >= 1) (data[source, 0], data[target, 0]) = (data[target, 0], data[source, 0]);
    }
}