using System.Diagnostics;
using System.Text;

namespace BNN;

public class Neuron
{
    private double[] _weights;
    private double _bias;
    private readonly double _momentum;
    private double[] _perWeightMomentums;
    private double _biasMomentum = 0.0;

    public Neuron(double[] weights, double bias, double momentum=0.0)
    {
        _weights = weights;
        _bias = bias;
        _momentum = momentum;
        _perWeightMomentums = new double[_weights.Length];
    }
    
    public double[] Weights => _weights;

    public double Bias => _bias;

    public double Apply(double[] inputs)
    {
        return inputs
            .Zip(_weights)
            .Select(x => x.First * x.Second).Sum() + _bias;
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
            errorToPropagate[i] = errorWrtNet * _weights[i];
            var weightDelta = _momentum * _perWeightMomentums[i] - (learningRate * pd_error_wrt_w_at_i);
            _perWeightMomentums[i] = weightDelta;

            newWeights[i] = _weights[i] + weightDelta;
            if (Double.IsNaN(newWeights[i]))
            {
                Debugger.Break();
                
                var sb = new StringBuilder();
                sb.AppendLine("neuron:" + i);
                sb.AppendLine("input:" + inputs[i]);
                sb.AppendLine("weight:" + _weights[i]);
                sb.AppendLine("newWeight:" + newWeights[i]);
                sb.AppendLine("weightDelta:" + weightDelta);
                sb.AppendLine("pd_error_wrt_w_at_i:" + pd_error_wrt_w_at_i);
                sb.AppendLine("errorWrtNet:" + errorWrtNet);
                sb.AppendLine("learningRate:" + learningRate);
                sb.AppendLine("perWeightMomentums:" + _perWeightMomentums[i]);
                sb.AppendLine("momentum:" + _momentum);
                sb.AppendLine("biasMomentum:" + _biasMomentum);
                sb.AppendLine("bias:" + _bias);
                    
                throw new ApplicationException($"training failed. {sb}");
            }
        }

        var biasDelta = _momentum * _biasMomentum - (learningRate * errorWrtNet);
        _biasMomentum = biasDelta;
        var newBias = _bias + biasDelta;
        if (Double.IsNaN(newBias))
        {
            throw new ApplicationException($"training failed. bias:{_bias}, newBias:{newBias}, biasDelta:{biasDelta}");
        }

        // activate the changes
        _weights = newWeights;
        _bias = newBias;
        
        return errorToPropagate;
    }
}

