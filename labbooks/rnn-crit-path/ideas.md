From [the Reddit thread](https://www.reddit.com/r/MachineLearning/comments/3j9sdj/experiment_log_matching_gru_performance_with/)
- benanne [links a paper](http://arxiv.org/abs/1312.6026) that's similar to this work. Models (2b) and (2b\*) are very similar to the ones I'm looking at. They've reminded me that [one experiment I carried out but didn't follow up on](https://github.com/andyljones/char-rnn-experiments/commit/7659f1f6) was the importance of feeding the inputs to second inter-step layer. My first thought is that it's important because it makes it easier to compute bilinear functions of the input, which might make gating functionality easier to learn. I've got no evidence for this right now though.
- benanne, mostly_reasonable and solus1232 make the point that the improvements might come from the simple increase in the number of parameters. My intuition is that it doesn't since the GRU has more params than any of the simple RNNs, but an experiment addressing this would be useful.
- derRoller asks whether additional GRU layers between timesteps would also improve performance. I'd guess that it does, but for my current matching-GRUs-with-RNN investigation it isn't a priority to look in to.

On investigating memory
- I'd like a more specific experiment on network memory. My hypothesis is that multiple in-time layers allow the simple RNN to remember things like the GRU does, but the Shakespeare task doesn't explicitly show this. Something like parenthesis matching would be good to look into. I think the NTM and stack-augmenting papers have several experiments along these lines.
- For the same reason, I think very small networks with only a handful of cells would be worth looking into. How does the recall capacity of RNNs scale with their size? In sufficiently small networks, could we effectively visualize it learning to memorize?
- Karpathy's RNN visualization work might be useful in demonstrating that the simple RNN does/doesn't learn to reference long-past characters.

Other
- One big difference between the RNN and GRU experiments is that the RNN uses relu nonlinearities while the GRU uses tanh. This is somewhat justified by the fact that I've never seen a GRU use a relu, but [Lyu & Zhu](https://github.com/huashiyiqike/LSTM-MATLAB) suggest that LSTMs do better with 'steeper' nonlinearities. What happens if I replace the tanh/sigmoids with the piecewise linear nonlinearities they suggest?
- I should explore saturation. Vary the sequence length and the network size and see how the test error responds.
