local encoding = require 'encoding'
local storage = require 'storage'
local gru = require 'gru'
local usetools = require 'usetools'

function sample_int(temperature, potentials)
  local ints = torch.Tensor(potentials:size(1))
  local relative_probs = torch.exp(potentials/temperature)
  local probs = relative_probs/torch.sum(relative_probs)
  return torch.multinomial(probs, 1):resize(1, -1)
end

function sample_char(alphabet, temperature, potentials)
  return encoding.ints_to_chars(alphabet, sample_int(temperature, potentials))
end

function sample(model, alphabet, cue, count, temperature)
  local forward, _ = usetools.make_forward_backward(model, 1)
  local states = {torch.zeros(1, model.config.n_neurons)}
  for i = 1, #cue - 1 do
    local input = encoding.chars_to_one_hot(alphabet, cue:sub(i,i)):view(1, 1, -1)
    _, _, states = forward(input, {states[1]})
  end

  local outputs = cue
  local potentials = {}
  for i = 1, count do
    local input = encoding.chars_to_one_hot(alphabet, outputs:sub(-1,-1)):view(1, 1, -1)
    potentials, _, states = forward(input, {states[1]})
    outputs = outputs .. sample_char(alphabet, temperature, potentials[1])
  end

  print(outputs)
end

local model, alphabet, options = storage.load_model('2015-08-30 21-06-06')
local cue = 'We are accounted poor citizens, '
sample(model, alphabet, cue, 200, 1.)
