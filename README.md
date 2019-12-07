## Multiple Instance Learning using Attention Mechanism

The project descrives benchmark accuracies for attention based multiple learning

### Objective

Classify a bag of MNIST images (~30 images per bag). The bags are of two classes - 1s and 0s. If a bag contains atleast 1 '9', the class is 1. If a bag contains no '9's, the class is zero.
The bags can be visualized as one large image of size 840 * 840 (each MNIST image is of size 28 * 28. 30 * 28 = 840), where this large image is divided into chunks or patches of 30 * 30.
Thus, the bags are weakly annotated. Meaning, we do not exactly know the coordinates of '9's, but we do know that it is present. Using attention, we are able to assign probability score of each patch being '9'.Also, these scores are used for final classification of the bag


### Model Architecture

Patch Classifier > Attention Module > Final Classifier

### References

> <https://arxiv.org/pdf/1802.04712v4.pdf>
