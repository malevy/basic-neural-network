namespace BNN;

public static class ListExtensions
{
    public static void Shuffle<T>(this IList<T> list)
    {
        var rand = new Random();
        var n = list.Count - 1;
        while (n > 1)
        {
            var k = rand.Next(n + 1);
            (list[n], list[k]) = (list[k], list[n]);
            n--;
        }
        
    }
}