namespace BNN;

public static class RandomExtensions
{
    public static double NextDouble(this Random @this, double min, double max)
    {
        return @this.NextDouble() * (max - min) + min;
    }

    /**
     * return a random item from the given list
     */
    public static T Random<T>(this Random @this, T[] items)
    {
        return items[@this.Next(items.Length)];
    }
}