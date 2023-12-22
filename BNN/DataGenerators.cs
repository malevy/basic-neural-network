namespace BNN;
using Plotly.NET;


public static class DataGenerators
{
    /*
     * based on https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/vertical.py
     * for any given sample n
     * result[n,0] = X coord
     * result[n,1] = Y coord
     * result[n,2] = the class (ie dataset) the point belongs to
     */
    public static double[,] BuildVerticalDataSet(int sampleCount, int classCount)
    {
        var rand = new Random();
        double[,] results = new double[sampleCount * classCount, 3];

        for (var c = 0; c < classCount; c++)
        {
            var offset = c * sampleCount;
            for (var n = 0; n < sampleCount; n++)
            {
                var X = rand.Randn(sampleCount).Select(v => (v * 0.1) + (c / 3.0)).ToArray();
                var Y = rand.Randn(sampleCount).Select(v => (v * 0.1) + 0.5).ToArray();
                results[offset + n, 0] = X[n];
                results[offset + n, 1] = Y[n];
                results[offset + n, 2] = c;
            }
        }

        return results;
    }

/*
* var data = DataGenerators.BuildVerticalDataSet(50,3);

var series0 = Enumerable.Range(0, data.GetLength(0))
    .Where(index => data[index, 2] == 0)
    .Select(index => new Tuple<double,double>(data[index, 0], data[index, 1]))
    .ToArray();

var series1 = Enumerable.Range(0, data.GetLength(0))
    .Where(index => data[index, 2] == 1)
    .Select(index => new Tuple<double,double>(data[index, 0], data[index, 1]))
    .ToArray();

var series2 = Enumerable.Range(0, data.GetLength(0))
    .Where(index => data[index, 2] == 2)
    .Select(index => new Tuple<double,double>(data[index, 0], data[index, 1]))
    .ToArray();

var charts = new[]
{
    Chart2D.Chart.Scatter<double, double, string>(
        xy: series0, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red")
        ),
    Chart2D.Chart.Scatter<double, double, string>(
        xy: series1, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("green")
    ),
    Chart2D.Chart.Scatter<double, double, string>(
        xy: series2, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("blue")
    )
};
Chart.Combine(charts).Show();
*/

    /*
     * for any given sample n
     * result[n,0] = X coord
     * result[n,1] = Y coord
     */
    public static double[,] BuildSineData(int numberOfSamples)
    {
        var rand = new Random();

        var x = new double[numberOfSamples];
        for (var j = 0; j < numberOfSamples; j++)
        {
            x[j] = rand.NextDouble();
        }
        Array.Sort(x);
        
        var samples = new double[numberOfSamples,2];
        for(var i=0; i<numberOfSamples; i++)
        {
            samples[i,0] = x[i]; // x
            samples[i,1] = Math.Sin(samples[i,0] * 2 * Math.PI); // y
        }
        
        return samples;
    }

}