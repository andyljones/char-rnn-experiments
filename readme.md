This project is intended to investigate better initialization schemes for character-level RNNs. The inspiration is

 - [Karpathy's char-rnn work](https://github.com/karpathy/char-rnn/), which relies on LSTMs
 - Le & Hinton's [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](http://arxiv.org/abs/1504.00941), which adjusts the initialization scheme of a RNN to achieve LSTM-like performance.
 - Mikolov et al.'s [Learning Longer Memory in Recurrent Neural Networks](http://arxiv.org/abs/1412.7753), which proposes a constraint on the RNN update matrix that also delivers LSTM-like performance.
 -  Saxe et al.'s [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120), aka the orthogonal initialization paper.