Did a fair bit of playing around today with identity initialization. All negative results.

 - First up was identity initialization ([commit](https://github.com/andyljones/char-rnn-experiments/commit/688e010c)). Wasn't any noticeably better than orthogonal initialization, and might have been slightly worse (I terminated early).
 - I wondered whether identity initialization might be a special case of orthogonal initialization. When you have two layers between timesteps in a simple RNN, could it be useful to initialize the orthogonal matrices of each layer as inverses of eachother? They'd combine to form an identity matrix across the timestep. Again, no better than plain orthogonal initialization ([commit](https://github.com/andyljones/char-rnn-experiments/commit/5dd41cc0)).
 - What about initializing the char-to-hidden and hidden-to-char layers with the identity, rather than with Glorot initialization? Nope, came out substantially worse ([commit](https://github.com/andyljones/char-rnn-experiments/commit/94619ac5)).
 - Tried initializing the recurrent weight matrix to be orthogonal, but with a region then set to the identity ([commit](https://github.com/andyljones/char-rnn-experiments/commit/86b528eb)). Again, no improvement.

Last thing was to check whether normalizing the character features to have mean 0 and standard deviation 1 would make any difference. Nope ([commit]((https://github.com/andyljones/char-rnn-experiments/commit/8640b04a)).

All in all: orthogonal initialization is feeling pretty darned good.
