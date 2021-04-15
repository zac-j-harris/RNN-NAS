# RNN-NAS


Neural Architecture Search for Recurrent Neural Nets.

Specifically, using a simple Genetic Algorithm to evolve unidirectional, bidirectional, and cascaded LSTM networks in order to find the optimal architecture.

Current running parameters:
- Dataset: CIFAR-10
- GA generations: 300
- Population size: 150
- Mutation rate: 0.3
- Elitism rate: 0.1
- Structure rate: 0.1


TODO:
- Introduce RS as a benchmark against which to compare
- Introduce GD as a benchmark...
- Modify NAS algorithm to attempt greater accuracies
