# Weights
AE1: Regular autoencoder trained on MNIST, has a dimension of 10
OTMap1: Optimal Transport Mapper from R^10 to R^10. Uses mae in both the loss function as well as the Optimal Transport algorithm. Five layers, and LeakyReLUs
ot_map_inputs.npy: The random inputs in the Optimal Transport algorithm.
ot_map_answers.npy: The answers according to the Optimal Transport mapping.
