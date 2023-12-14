namespace BNN;

public static class ArrayUtils
{
    // return the index of the max value in the array
    public static int ArgMax(double[] arr)
    {
        return Array.IndexOf(arr, arr.Max());
    } 
}