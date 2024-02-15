from mnistSimpleCNN import train, test
from pyrfd import RFDSqExp

def test_mnist():
    train.run(RFDSqExp, p_epochs=2)
    accuracy = test.run()
    assert accuracy > 0.98
