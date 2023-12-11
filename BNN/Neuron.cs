using System.Text;

namespace BNN;

public class Neuron
{
    private double[] _weights;
    private double _bias;
    private readonly double _momentum = 0.9;
    private double[] _perWeightMomentums;
    private double _biasMomentum = 0.0;

    public Neuron(double[] weights, double bias)
    {
        _weights = weights;
        _bias = bias;
        _perWeightMomentums = new double[_weights.Length];
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
        sb.Append("\"weights\":[");
        sb.Append(weights);
        sb.Append("],\"bias\":");
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
            var weightDelta = _momentum * _perWeightMomentums[i] - (learningRate * pd_error_wrt_w_at_i);
            _perWeightMomentums[i] = weightDelta;

            newWeights[i] = _weights[i] + weightDelta;
        }

        var biasDelta = _momentum * _biasMomentum - (learningRate * errorWrtNet);
        _biasMomentum = biasDelta;
        var newBias = _bias + biasDelta;

        // activate the changes
        _weights = newWeights;
        _bias = newBias;
        
        return errorToPropagate;
    }
}

