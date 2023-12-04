namespace BNN;

/**
 * A single-layer neural network can only be used to represent linearly separable functions.
 * This means very simple problems where, say, the two classes in a classification problem
 * can be neatly separated by a line.
 * in most cases, for a NN to fit a non-linear function (convex region), it will need at
 * least 2 hidden layers and those layers will need a non-linear activation function
 */
public class NetworkBuilder
{
    private record LayerDesign(int Neurons, IActivationFunction ActivationFunction);
    
    private readonly int _inputs;
    private readonly IList<LayerDesign> _layersDesigns = new List<LayerDesign>();
    private Func<double[], double[], double> _aggregateErrorFunction;
    private Func<double[], double[], double[]> _gradientErrorFunction;


    private NetworkBuilder(int inputs)
    {
        _inputs = inputs;
    }

    public NetworkBuilder WithLayer(int neuronCount, IActivationFunction activationFunction)
    {
        _layersDesigns.Add(new LayerDesign(neuronCount, activationFunction));
        return this;
    }

    public Network Build()
    {
        if (!_layersDesigns.Any()) throw new InvalidOperationException("no layers were specific");

        var inputs = _inputs;
        var layers = new List<Layer>();
        foreach (var ld in _layersDesigns)
        {
            layers.Add(new Layer(inputs, ld.Neurons, ld.ActivationFunction));
            
            // the number of neurons becomes the number of inputs for the next layer
            inputs = ld.Neurons;
        }

        return new Network(
            layers.ToArray(), 
            _aggregateErrorFunction, 
            _gradientErrorFunction);
    }

    public static NetworkBuilder WithInputs(int inputs)
    {
        return new NetworkBuilder(inputs);
    }

    public NetworkBuilder WithAggregateLossFunction(Func<double[], double[], double> func)
    {
        _aggregateErrorFunction = func;
        return this;
    }

    public NetworkBuilder WithGradientLossFunction(Func<double[], double[], double[]> func)
    {
        _gradientErrorFunction = func;
        return this;
    }
}