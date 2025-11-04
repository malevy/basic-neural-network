
using BNN;
using Plotly.NET;

// var data = DataGenerators.BuildSineData(20);

 // ArrayUtils.Shuffle(data);
 //
 // var series0 = Enumerable.Range(0, data.GetLength(0))
 //  .Select(index => new Tuple<double,double>(data[index, 0], data[index, 1]))
 //  .ToArray();
 //
 // var charts = new[]
 // {
 //  Chart2D.Chart.Scatter<double, double, string>(
 //   xy: series0, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red")
 //  ),
 // };
 // Chart.Combine(charts).Show();

 var data = DataGenerators.BuildVerticalDataSet(100,3);

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
