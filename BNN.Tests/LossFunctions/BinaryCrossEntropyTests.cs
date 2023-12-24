namespace BNN.Tests.LossFunctions;

public class BinaryCrossEntropyTests
{
    [Test]
    public void MatchMathWorks()
    {
        var loss = BNN.LossFunctions.BinaryCrossEntropy(new[] { 1.0 }, new[] { 0.79 });
        Assert.That(loss, Is.EqualTo(0.23575).Within(0.001));
    }

    [Test]
    public void NonMatchMathWorks()
    {
        var loss = BNN.LossFunctions.BinaryCrossEntropy(new[] { 0.0 }, new[] { 0.19 });
        Assert.That(loss, Is.EqualTo(0.21075).Within(0.001));
    }

    [Test]
    public void MatchDerivativeMathWorks()
    {
        var gradient = BNN.LossFunctions.BinaryCrossEntropyDerivative(new[] { 1.0 }, new[] { 0.79 });
        Assert.That(gradient[0], Is.EqualTo(-1.26585).Within(0.001));
    }
    
    [Test]
    public void NonMatchDerivativeMathWorks()
    {
        var gradient = BNN.LossFunctions.BinaryCrossEntropyDerivative(new[] { 0.0 }, new[] { 0.19 });
        Assert.That(gradient[0], Is.EqualTo(1.2345).Within(0.001));
    }
    
    
}