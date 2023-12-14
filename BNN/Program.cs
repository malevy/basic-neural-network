
using BNN;
using Plotly.NET;

// SingleNeuron.Run();
 // XorTest.Run();
//VerticalTest.Run();
 var data = DataGenerators.BuildSineData(20);

 ArrayUtils.Shuffle(data);
 
 var series0 = Enumerable.Range(0, data.GetLength(0))
  .Select(index => new Tuple<double,double>(data[index, 0], data[index, 1]))
  .ToArray();

 var charts = new[]
 {
  Chart2D.Chart.Scatter<double, double, string>(
   xy: series0, mode: StyleParam.Mode.Markers, MarkerColor: Color.fromString("red")
  ),
 };
 Chart.Combine(charts).Show();



