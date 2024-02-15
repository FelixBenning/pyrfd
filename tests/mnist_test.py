from mnistSimpleCNN import train, test
from pyrfd import RFD

def test_mnist():
    train.run(RFD, p_epochs=2)
    accuracy = test.run()
    assert accuracy > 0.98
