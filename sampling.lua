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

function trim_and_pad(str, length, pad_char)
  if #str < length then
    return string.rep(pad_char, length - #str) .. str
  elseif #str == length then
    return str
  else
    return str:sub(-length, -1)
  end
end

function sample(model, n_timesteps, alphabet, cue, count)
  local results = trim_and_pad(cue, n_timesteps, ' ')
  for i = 1, count do
    local input = results:sub(-n_timesteps, -1)
    local input = encoding.chars_to_one_hot(alphabet, input)
    local input = input:view(1, input:size(1), input:size(2))
    local output, _ = unpack(model:forward({input, model.default_state[{{1}}]}))
    local result = encoding.one_hot_to_chars(alphabet, output[1])
    results = results .. result:sub(-1, -1)
  end

  print(results)
  return results
end

local model, alphabet, options = storage.load_model('2015-08-29 15-45-15')
sample(model, options.n_timesteps, alphabet, 'Before we proceed any further, he', 100)
