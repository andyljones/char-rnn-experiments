Following up on what [some folks on reddit said](https://www.reddit.com/r/MachineLearning/comments/3j9sdj/experiment_log_matching_gru_performance_with/), I've run some experiments comparing RNNs to GRUs where the number of parameters is kept constant (rather than the hidden state size). These results should be contrasted against those in [post \#1](https://github.com/andyljones/char-rnn-experiments/blob/master/labbooks/rnn-crit-path/post-1.md)

 - Baseline is a 128-cell GRU. Test loss: 1.49 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/6367755fc5d78f3710de113dcc6a1baa46e6f3c8))
 - Simple RNN with 1 layer per timestep and 230 cells: 1.49 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/e5f746c65037ed74c0ef98ba5afdc0030c79d4ee))
 - Simple RNN with 2 layers per timestep and 160 cells: 1.49 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/5d2816ad4fb63a60f785422c5d8f824962ec03c5))
 - Simple RNN with 3 layers per timestep and 128 cells: 1.49 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/6367755fc5d78f3710de113dcc6a1baa46e6f3c8))

Several things to note here

 - First is the performance of the 3-layer simple RNN. I ran what I thought was the same experiment yesterday and (as reported in post \#1) got a test error of 1.46. I think this might be an indication that my test set size is too small.
 - Second is that apparently the architecture - single-layer, multi-layer, RNN or GRU - makes no difference at all when the param count is fixed (to 83k, which is what the GRU experiment had in yesterday's work). What I'm thinking now is that maybe generative language modelling is a bad way to benchmark these architectures, at least when the networks are so small: it could be that the networks all sink their resources in remembering very short-term dependencies (since spelling words correctly probably gets you a lot of objective for your resources), while longer-term dependencies are ignored.

 Tomorrow I think I'll start exploring other, simpler, tasks. 
