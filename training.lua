local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'

function decode(alphabet, batch)
  local results = {}
  for i = 1, batch:size(1) do
    results[i] = encoding.one_hot_to_chars(alphabet, batch[i])
  end

  return results
end

function reshape(batch)
  local sequences = batch:transpose(1, 2):split(1, 1)
  for i = 1, #sequences do
    sequences[i] = sequences[i]:clone():view(batch:size(1), batch:size(3))
  end
  return sequences
end

local n_neurons, n_timesteps, n_samples = 20, 10, 2

local text = batcher.load_text()
local alphabet, batch_iterator = batcher.make_batch_iterators(text, torch.Tensor{1}, n_timesteps, n_samples)

local model = gru.build(n_timesteps, table.size(alphabet), n_neurons)
local initial_state = torch.zeros(n_samples, n_neurons)
local batch = reshape(batch_iterator[1]())
local output, final_state = unpack(model:forward({batch, initial_state}))
for _, v in pairs(decode(alphabet, output)) do print(v) end
