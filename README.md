# gated-feedback-rnn
a practical implementation of Gated Feedback RNN: https://arxiv.org/abs/1502.02367

While serving the purpose of exploring the idea from the paper, the major goal of this project is to implement the gated feedback rnn as close to other RNN implemention in Tensorflow's source code as possible. It should make the best use of what Tensor has to offer, and can be integrated into normal procedure of running an RNN in Tensorflow seamlessly, e.g., running with `tf.nn.dynamic_run`. 
