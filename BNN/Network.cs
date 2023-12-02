using System.Diagnostics;
using System.Text;

namespace BNN;

public class Network
{
    private readonly Layer[] _layers;
    
    public Network(Layer[] layers)
    {
        _layers = layers;
    }
    
    public double[] Apply(double[] inputs)
    {
        var outputs = inputs;
        foreach (var layer in _layers)
        {
            outputs = layer.Apply(outputs);
        }
        
        return outputs;
    }

    public string Dump()
    {
        var sb = new StringBuilder();
        sb.Append("{layers:[");
        foreach (var layer in _layers)
        {
            sb.Append(layer.Dump());
            sb.Append(',');
        }
        sb.AppendLine("]}");
        return sb.ToString();
    }
    
    public double Train(double[] inputs, double[] targets, double learningRate)
    {
        // the original inputs are at index 0;
        var inputsList = new List<double[]> {inputs};

        var outputs = inputs;
        foreach (var layer in _layers)
        {
            outputs = layer.Apply(outputs);
            inputsList.Add(outputs);
        }

        // the final predicted values end up as the last entry
        // in inputList.
        var predicted = inputsList.Last();
        inputsList.Remove(predicted);

        Debug.Assert(predicted.Length == targets.Length);
        
        // calculate the error using the loss function
        // TODO - the loss function should be supplied as a parameter
        var averageError = predicted
            .Select((p, i) => LossFunctions.AbsoluteError(targets[i], p))
            .Sum();

        var errorsWrtOutput = predicted
            .Select((p, i) => LossFunctions.dSquaredError(targets[i], p))
            .ToArray();

        // propagate the error backwards through the layers
        var errorsToPropagate = errorsWrtOutput;
        for (var i = inputsList.Count-1; i >= 0; i--)
        {
            errorsToPropagate = _layers[i].BackProp(inputsList[i], errorsToPropagate, learningRate);
        }

        // return the average error prior to adjusting weights
        return averageError;
    }
}