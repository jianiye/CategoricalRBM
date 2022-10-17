# CategoricalRBM

### This project builds a categorical restricted boltzmann machine with Pytorch.

A restricted boltzmann machine(RBM) consists of a visible layer and a hidden layer. The two layers are fully connected. The weight tensors can be considered as energy transformation between the two layers.
The system is in equillibirum state if the energy flow between the two layers are balanced. This also means the information flow is in equillibrium, in that state we can reconstruct visible through the weight tensors.

The euillibrium state corresponds to a minimum free energy, so an RBM uses free energy $F(v)$ as the loss metric.

For an input Visible tensor, the weight tensors forward it to the hidden tensor, the hidden tensor backwards with the shared weight tensor to a reconstructed visible tensor.

Usually, a visible node is a bounoulli distributed variable, but in CategoricalRBM, it is a one-hot tensor, therefore the model enables categorical classification.

<img width="558" alt="image" src="https://user-images.githubusercontent.com/115207895/196232177-0631f242-5b06-45b9-82a4-724fd7d787a4.png">

#### Training with the Movie-Lens-1M data

The Movie-Lens-1M data is a sparce $N_{user}*N_{movie}$ tensor, each non-zero element has value ~[1,2,3,4,5]. The traning takes batched user tensor of size $N_{movie}*5$ as input, the zeroed values are masked.
