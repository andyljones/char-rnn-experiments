This is/will be a repository of experiments on RNN initialization.

Right now though it's mostly a ground-up rewrite of Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) project. The main difference is that Karpathy instantiates a different computation graph for each timestep and manipulates them individually; this version uses a single computation graph, which feels easier to deal with. 
