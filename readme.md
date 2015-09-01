This is a ground-up rewrite of Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) project intended to serve as a foundation for my experiments with RNNs. I rewrote it rather than use the original because I wanted to know how every.single.part of it fit together before I started playing around.

To run most of the experiments, simply check out the relevant commit and (for now at least) run `training.lua`. You'll need to have Torch installed, along with the `nn`, `nngraph` and `optim` packages. Instructions on how to install those can be  
found in the [char-rnn readme](https://github.com/karpathy/char-rnn).
