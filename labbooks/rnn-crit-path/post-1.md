I'm currently using Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) project to experiment with RNNs. If you haven't come across it before, the idea in char-rnn is to use a recurrent neural network to predict each character in a piece of text from preceding characters. What's great about char-rnn as a platform for investigating RNNs is that it's small and self-contained - you don't need a database of word vectors or a GPU to do something interesting.

## GRUs vs simple RNNs

Today's line of investigation is about GRUs. [GRU](http://arxiv.org/abs/1502.02367)s are RNNs with a specific 'memory cell' architecture that's intended to let the network remember things over many timesteps. They (and the older, more complex LSTMs) have improved over the performance of simple RNNs on a variety of tasks.

What I've been wondering though is if this architecture is so powerful, why don't simple RNNs pick up on it during training[\*, \*\*]? I came up with the hypothesis that maybe the problem was that in a plain RNN, there's only one learnable transformation applied between successive timesteps, while in a GRU there are two: one to calculate the value of the reset gate, then another that takes that value and comes up with a new candidate cell state. Effectively, a GRU gets twice as long to think about things as an RNN.

So, experiment: what happens if we put multiple layers between each timestep of a simple RNN? The second layer gets the same character input, plus the output of the first layer. Using the Shakespeare task that comes with char-rnn,

 - The GRU achieves a test loss of 1.50 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/023fd6addcf815cea3b2d0d10619be60d12a9db5))
 - The 1-layer simple RNN achieves a test loss of 1.58 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/9841931592e6ae8255d119b00412db1bc7608235))
 - The 2-layer simple RNN achieves a test loss of 1.50 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/8d1086b879eb9da4eeaf298958fddf350c5407aa))
 -  The 3-layer simple RNN achieves a test loss of 1.47 ([commit](https://github.com/andyljones/char-rnn-experiments/commit/6d610294fd6459c4577fb5f4afbb997530ebb172))

(lower is better, and while the number of parameters varies, the size of the hidden state carried between timesteps is the same in each one)

That's in no way conclusive - it's a single task and a rather contrived one at that - but it's enough to keep me interested.

If you're curious about the details of the experiments, the code for each one is linked. More generally, every experiment I've carried out can be found on the [experiment/rnn-crit-path](https://github.com/andyljones/char-rnn-experiments/tree/experiment/rnn-crit-path) branch. Two points might be of particular interest: that the simple RNNs use ReLU nonlinearities, as suggested by [Le et al](http://arxiv.org/abs/1504.00941), and that initialization is done using [orthogonal initialization](http://arxiv.org/abs/1312.6120) for square matricies and [Glorot initialization](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) for the non-square ones. I like these schemes because they reduce the number of hyperparameters I have to pick out of a hat.

As a side note, these are by no means the best results you can achieve on the Shakespeare task. The networks are deliberately small and simple so that the experiments are quick to run.

## Open Computer Science
It might seem little strange to talk openly about a line of inquiry that's in its infancy and (if previous experience has anything to go by) has a 90% chance of being a dead end. Thing is, I think there's value in documenting what I do even if it *does* lead to nowhere. Someone will probably have a similar idea to this one down the line, and Googling up this post will save them a whole bunch of time.  

The other thing I'm interested in doing is using Git as a labbook. By committing after every experiment and sticking the results in the commit comment, I'm hoping I'll be able to build a rigorous, easily-searchable log of everything I've done.

Anyway, I'll continue to push to the repo and update here as I learn more, and we'll see what happens.

------------------------------------

\* I know there's a theorem of Bengio's (?) that says simple RNNs can't remember a bit perfectly and still be robust against noise, but I haven't read that paper so it didn't interrupt my thought train.

\*\* Actually, my original thought\*\*\* was whether I could hand-code the relations that define a GRU into the initialization of a simple RNN. But seeing if the training alg could find something on its own felt like it'd be a lot easier.

\*\*\* This itself was founded in some thinking about [Le et al's](http://arxiv.org/abs/1504.00941) neat paper on initializing simple RNNs with the identity matrix, which gives another way to bring simple RNNs up to LSTM-ish performance.
