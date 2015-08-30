local encoding = require 'encoding'
local storage = require 'storage'
local gru = require 'gru'

function decode(alphabet, batch)
  local results = {}
  for i = 1, batch:size(1) do
    results[i] = encoding.one_hot_to_chars(alphabet, batch[i])
  end

  return results
end

function probabilistic_one_hot_to_ints(temperature, one_hot)
  local ints = torch.Tensor(one_hot:size(1))
  for i = 1, one_hot:size(1) do
    local relative_probs = torch.exp(one_hot[i]/temperature)
    local probs = relative_probs/torch.sum(relative_probs)
    ints[i] = torch.multinomial(probs, 1):resize(1)
  end
  return ints
end

function probabilistic_one_hot_to_chars(alphabet, temperature, one_hot)
  return encoding.ints_to_chars(alphabet, probabilistic_one_hot_to_ints(temperature, one_hot))
end

function trim_and_pad(str, length, pad_char)
  if #str < length then
    return string.rep(pad_char, length - #str) .. str
  elseif #str == length then
    return str
  else
    return str:sub(-length, -1)
  end
end

function sample(model, n_timesteps, alphabet, cue, count, temperature)
  local results = trim_and_pad(cue, n_timesteps, ' ')
  for i = 1, count do
    local input = results:sub(-n_timesteps+1, -1)
    local input = encoding.chars_to_one_hot(alphabet, input)
    local input = input:view(1, input:size(1), input:size(2))
    local output, _ = unpack(model:forward({input, model.default_state[{{1}}]}))
    local result = probabilistic_one_hot_to_chars(alphabet, temperature, output[1])
    results = results .. result:sub(-1, -1)
  end

  print(results)
  return results
end

local model, alphabet, options = storage.load_model('2015-08-30 10-11-13')
local cue = 'We are accounted poor citizens, the patricians good.\nWhat authority surfeits on would relieve us: if they'
sample(model, options.n_timesteps, alphabet, cue, 200, 0.8)
