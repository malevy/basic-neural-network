using System.Text;

namespace BNN;

public class Layer
{
    private readonly Neuron[] _neurons;
    
    public Layer(int inputCount, int neuronCount, IActivationFunction actFunc)
    {
        var weightLimit = WeightScale(inputCount, neuronCount);
        var rand = new Random();

        _neurons = Enumerable.Range(0, neuronCount)
            .Select(_ => WeightFactory(inputCount))
            .Select(weights => new Neuron(weights, rand.NextDouble(), actFunc))
            .ToArray();
        
        return;

        double[] WeightFactory(int cnt) =>
            Enumerable.Range(0, cnt)
                .Select(_ => 0.1 * rand.NextDouble(-weightLimit, weightLimit))
                .ToArray();
    }

    public double[] Apply(double[] inputs)
    {
        var outputs = new double[this._neurons.Length];
        for (var i = 0; i < _neurons.Length; i++)
        {
            outputs[i] = _neurons[i].Apply(inputs);
        }
        return outputs;
    }

    public String Dump()
    {
        var sb = new StringBuilder();
        sb.Append("{neurons:[");
        foreach (var neuron in _neurons)
        {
            sb.Append( neuron.Dump());
        }
        sb.AppendLine("]}");
        return sb.ToString();
    }
    
    /**
     * Adjusts the neurons and returns the proportioned error for each input
     * by weight
     */
    public double[] BackProp(double[] inputs, double[] errorWrtOutput, double learningRate)
    {
        var aggregateErrorPerInput = new double[inputs.Length];
        for (var i = 0; i < _neurons.Length; i++)
        {
            var errorPerInput = _neurons[i].BackProp(inputs, errorWrtOutput[i], learningRate);
            
            for (var j = 0; j < inputs.Length; j++)
            {
                aggregateErrorPerInput[j] += errorPerInput[j];
            }
        }

        return aggregateErrorPerInput;
    }

    /**
     * Glorot Uniform Initializer
     * Glorot & Bengio, AISTATS 2010
     * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
     */
    private static double WeightScale(int inputCount, int outputCount)
    {
        return Math.Sqrt(6.0 / (inputCount + outputCount));
    }
}