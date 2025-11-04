using System.Diagnostics;

namespace BNN.Tests;

using Microsoft.VisualBasic.FileIO;

/// <summary>
/// building a model for the Kaggle Titanic problem
/// https://www.kaggle.com/competitions/titanic
///
/// very little anaysis of the data and feature engineering was done.
/// </summary>
public class TitanicTest
{
    public static void Run()
    {
        var network = NetworkBuilder.WithInputs(7)
            .WithLayer(14, new ActivationFunctions.ReLuFunction(), 0.7)
            .WithLayer(1, new ActivationFunctions.SigmoidFunction())
            .WithAggregateLossFunction(LossFunctions.BinaryCrossEntropy)
            .WithGradientLossFunction(LossFunctions.BinaryCrossEntropyDerivative)
            .Build();
        
        var data = ReadTrainingData();
        Console.WriteLine($"read {data.Count} lines");
        var training = data.Take((int)(data.Count*.8)).OrderBy(s => s[8]).ToList();
        var validation = data.Skip((int)(data.Count*.8)).ToList();
        
        var learningRate = new LearningRate(3e-3, 1e-6);
        List<double> errors = new();
        
        // train
        var err = 0.0;
        var expected = new double[1];
        var inputs = new double[7];
        for (var epoch = 0; epoch < 250; epoch++)
        {
            for (var n = 0; n < training.Count; n++)
            {
                var sample = training[n];
                Array.Copy(sample, 1, inputs, 0, 7);
                expected[0] = sample[8];
                err = network.Train(inputs, expected, learningRate.Value);
            }

            if (epoch % 10 == 0)
            {
                Console.WriteLine($"epoch:{epoch} error:{err} lr:{learningRate.Value}");
                errors.Add(err);
            }

            learningRate.Decay();
        }

        Console.WriteLine($"last error:{err}");
        errors.Add(err);
        ErrorGraph.Graph(errors);

        // validate
        var correct = 0.0;
        foreach (var sample in validation)
        {
            Array.Copy(sample, 1, inputs, 0, 7);
            var prediction = network.Apply(inputs);
            Console.Write($"actual:{sample[8]:0.00} predicted:{prediction[0]:0.00} ");
            if (Math.Abs(sample[8] - prediction[0]) < 0.4)
            {
                correct++;
                Console.WriteLine("\u221a");
            }
            else
            {
                Console.WriteLine();
            }
        }

        Console.WriteLine($"Accuracy: {correct} out of {validation.Count} ({correct / validation.Count:0.00})");
        
    }

    private static List<double[]> ReadTrainingData()
    {
        /* 
         * Position and data for each row read
         * 0: PassengerId
         * 1: Passenger Class (Pclass) [1,2,3] 
         * 2: Sex [0=male, 1=female]
         * 3: Age normalized between 0 and 1
         * 4: number of siblings (SibSp)
         * 5: number of parents (Parch)
         * 6: Log(Fare)
         * 7: where the passenger was Embarked [0,1,2]
         * 8: Survived
         */
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
                if (values == null) continue;

                // skip the lines with one or more infinity or negative infinaty
                if (values.Any(v => v.Contains("inf"))) continue;

                data.Add(values.Select(x => Double.Parse(x)).ToArray());
            }
            catch (MalformedLineException ex)
            {
                Console.WriteLine($"skipping line {ex.LineNumber}");
            }
            catch (FormatException)
            {
                Console.WriteLine($"cannot convert value to double on line {parser.LineNumber}");
            }
        }

        return data;
    }
}