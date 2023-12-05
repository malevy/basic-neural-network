using System.Diagnostics;
using System.Text;

namespace BNN;

public class Network
{
    private readonly Layer[] _layers;
    private Func<double[], double[], double> _aggregateErrorFunction;
    private Func<double[], double[], double[]> _gradientErrorFunction;
    
    /// <summary>
    /// Instantiate the Neural Network
    /// </summary>
    /// <param name="layers">An array of Layer object, each of which describe one layer
    /// in the network. The layers will be created in the other they appear in the given array
    /// </param>
    /// <param name="aggregateErrorFunction">
    /// A function that calculates the aggregate error.
    /// (target[],predicted[]) => aggregate error
    /// </param>
    /// <param name="gradientErrorFunction">
    /// A function that calculates the gradient errors.
    /// (target[],predicted[]) => gradient[]
    /// </param>
    public Network(
        Layer[] layers, 
        Func<double[], double[], double> aggregateErrorFunction, 
        Func<double[], double[], double[]> gradientErrorFunction)  
    {
        _layers = layers;
        _aggregateErrorFunction = aggregateErrorFunction;
        _gradientErrorFunction = gradientErrorFunction;
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

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="targets"></param>
    /// <param name="learningRate"></param>
    /// <returns></returns>
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
        var aggregateError = _aggregateErrorFunction.Invoke(targets, predicted);
        var errorsWrtOutput = _gradientErrorFunction.Invoke(targets, predicted); 

        // propagate the error backwards through the layers
        var errorsToPropagate = errorsWrtOutput;
        for (var i = inputsList.Count-1; i >= 0; i--)
        {
            errorsToPropagate = _layers[i].BackProp(inputsList[i], errorsToPropagate, learningRate);
        }

        // return the average error prior to adjusting weights
        return aggregateError;
    }
}