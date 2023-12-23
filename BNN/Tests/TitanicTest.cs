using System.Diagnostics;

namespace BNN.Tests;

using Microsoft.VisualBasic.FileIO;

public class TitanicTest
{
    public static void Run()
    {
        var trainingData = ReadTrainingData();
        Console.WriteLine($"read {trainingData.Count} lines");
    }

    private static List<double[]> ReadTrainingData()
    {
        var stream =
            new StreamReader(Path.Combine(Directory.GetCurrentDirectory(), "Tests/Content/train-scrubbed.csv"));
        if (stream == null) throw new InvalidOperationException("unable to open training data file");

        List<Double[]> data = new();
        using var parser = new TextFieldParser(stream);
        parser.SetDelimiters(",");
        var fields = parser.ReadFields(); // skip header

        while (!parser.EndOfData)
        {
            try
            {
                var values = parser.ReadFields();
                if (values is not { Length: 8 }) continue;
                    
                // skip the lines with one or more infinity or negative infinaty
                if (values.Any(v => v.Contains("inf"))) continue;
                    
                data.Add(values.Select(x => Double.Parse(x)).ToArray());
            }
            catch (MalformedLineException ex)
            {
                Console.WriteLine($"skipping line {ex.LineNumber}");
            }
            catch (FormatException fe)
            {
                Console.WriteLine($"cannot convert value to double on line {parser.LineNumber}");
            }
        }

        return data;
    }
}