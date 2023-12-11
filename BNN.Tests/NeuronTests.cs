namespace BNN.Tests;

public class NeuronTests
{
    [Test]
    public void ApplyTest()
    {
        var neuron = new Neuron(new[] { -3.0, -1.0, 2.0 }, 1.0);
        var inputs = new[]{1.0, -2.0, 3.0};
        var result = neuron.Apply(inputs);
        Assert.That(result, Is.EqualTo(6.0));
    }

    [Test]
    public void BackPropTest()
    {
        var neuron = new Neuron(new[] { -3.0, -1.0, 2.0 }, 1.0);
        var inputs = new[]{1.0, -2.0, 3.0};

        neuron.BackProp(inputs, 1.0, 0.001);
        
        Assert.That(neuron.Weights[0], Is.EqualTo(-3.001));
        Assert.That(neuron.Weights[1], Is.EqualTo(-0.998));
        Assert.That(neuron.Weights[2], Is.EqualTo(1.997));
        Assert.That(neuron.Bias, Is.EqualTo(0.999));

    }
    
}