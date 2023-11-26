using System.Text;

namespace BNN;

public class Neuron
{
    private double[] _weights;
    private double _bias;
    private readonly IActivationFunction _activationFunc;

    // these are required for training
    private double _net;
    private double _out;
    
    public Neuron(double[] weights, double bias, IActivationFunction activationFunction)
    {
        _weights = weights;
        _bias = bias;
        _activationFunc = activationFunction;
    }
    
    public double Apply(double[] inputs)
    {
        _net = inputs.Select((x, i) => x * _weights[i]).Sum() + _bias;
        _out = _activationFunc.Squash(_net);
        return _out;
    }

    public string Dump()
    {
        var sb = new StringBuilder();
        var weights = String.Join(",", _weights);
        sb.Append("{");
        sb.Append("weights:[");
        sb.Append(weights);
        sb.Append("],bias:");
        sb.Append(_bias);
        sb.AppendLine("}");
        return  sb.ToString();

    }
    
    public double[] BackProp(double[] inputs, double errorWrtOutput, double learningRate)
    {
        // the error property is errorWrtOutput
        
        // the partial derivative of the net wrt the weight[i] is the input[i]
        var pd_output_wrt_net = _activationFunc.PartialDee(_net, _out);

        var newWeights = new double[_weights.Length];
        var errorToPropagate = new double[_weights.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            var pd_error_wrt_w_at_i = errorWrtOutput * pd_output_wrt_net * inputs[i];
            errorToPropagate[i] = pd_error_wrt_w_at_i;
            newWeights[i] = _weights[i] - (learningRate * pd_error_wrt_w_at_i);
        }

        // TODO - verify that this is the correct way to update the bias
        var newBias = _bias - (learningRate * errorWrtOutput * pd_output_wrt_net);

        // activate the changes
        _weights = newWeights;
        _bias = newBias;
        
        return errorToPropagate;
    }
}

