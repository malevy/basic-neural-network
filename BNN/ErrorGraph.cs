using Plotly.NET.CSharp;

namespace BNN;
using Plotly.NET;

public static class ErrorGraph
{
    public static void Graph(IEnumerable<double> errors)
    {
        var errorsList = errors.ToList();
        var pointCount = errorsList.Count();
        Chart2D.Chart.Line<int, double, double>(
            x: Enumerable.Range(0, pointCount), 
            y: errorsList, 
            MarkerColor: Color.fromString("black")).Show();
    }
}