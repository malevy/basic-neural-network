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
    private record LayerDesign(int neurons, IActivationFunction ActivationFunction);
    
    private readonly int _inputs;
    private IList<LayerDesign> _layersDesigns = new List<LayerDesign>();

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
            layers.Add(new Layer(inputs, ld.neurons, ld.ActivationFunction));
            
            // the number neurons becomes the number of inputs for the next layer
            inputs = ld.neurons;
        }

        return new Network(layers.ToArray());
    }

    public static NetworkBuilder WithInputs(int inputs)
    {
        return new NetworkBuilder(inputs);
    }
}