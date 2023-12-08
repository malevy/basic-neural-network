namespace BNN.Tests.LossFunctions;

public class CategoricalCrossEntropyTests
{

    [Test]
    public void ErrorCalculation()
    {
        var predicted = new[] { 0.7, 0.1, 0.2 };
        var targets = new[] { 1.0, 0, 0 };

        var expected = 0.3566749439373245;

        var error = BNN.LossFunctions.CategoricalCrossEntropy(targets, predicted);
        Assert.That(error, Is.EqualTo(expected).Within(0.00001));

    }

    [Test]
    public void DerivativeCalculation()
    {
        var targets = new[] { 0, 1.0, 0, 0 };
        var predicted = new[] { 0.05, 0.85, 0.10, 0 };
        var expected = new[] { 0, -1.17647059, 0, 0 };
        var actual = BNN.LossFunctions.CategoricalCrossEntropyDerivative(targets, predicted);
        Assert.That(actual, Is.EqualTo(expected).Within(0.00001));
        
    }
    
}