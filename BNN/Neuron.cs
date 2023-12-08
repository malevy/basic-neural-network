using System.Text;

namespace BNN;

public class Neuron
{
    private double[] _weights;
    private double _bias;

    public Neuron(double[] weights, double bias)
    {
        _weights = weights;
        _bias = bias;
    }
    
    public double Apply(double[] inputs)
    {
        return inputs.Select((x, i) => x * _weights[i]).Sum() + _bias;
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
    
    public double[] BackProp(double[] inputs, double errorWrtNet, double learningRate)
    {
        
        var newWeights = new double[_weights.Length];
        var errorToPropagate = new double[_weights.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            var pd_error_wrt_w_at_i = errorWrtNet * inputs[i];
            errorToPropagate[i] = pd_error_wrt_w_at_i;
            newWeights[i] = _weights[i] - (learningRate * pd_error_wrt_w_at_i);
        }

        var newBias = _bias - (learningRate * errorWrtNet);

        // activate the changes
        _weights = newWeights;
        _bias = newBias;
        
        return errorToPropagate;
    }
}

