local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'

function batch_one_hot_to_ints(one_hots)
  local ints = torch.Tensor(one_hots:size(1), one_hots:size(2))
  for i = 1, one_hots:size(1) do
    ints[i] = encoding.one_hot_to_ints(one_hots[i])
  end
  return ints
end

function forward(modules, input, initial_state)
  local batch_size, seq_length, n_symbols = unpack(torch.totable(input:size()))

  local states = torch.Tensor(batch_size, seq_length, initial_state:size(2))
  states[{{}, 1}] = initial_state

  local outputs = torch.Tensor(batch_size, seq_length - 1, n_symbols)
  local targets = batch_one_hot_to_ints(input)
  local loss = 0

  for i = 1, seq_length - 1 do
    local output, state = unpack(modules[i]:forward{input[{{}, i}], states[{{}, i}]})
    outputs[{{}, i}] = output
    states[{{}, i + 1}] = state

    loss = loss + modules[i].criterion:forward(output, targets[{{}, i+1}])
  end

  return outputs, states, loss/(seq_length - 1)
end

function backward(modules, inputs, outputs, final_state)
  local batch_size, seq_length, n_symbols = unpack(torch.totable(input:size()))

  local states = torch.Tensor(batch_size, seq_length, final_state:size(2))
  states[{{}, seq_length - 1}] = final_state

  local targets = batch_one_hot_to_ints(input)

  for i = seq_length - 1, 1, -1 do
    local loss_grad = modules[i].criterion:backward(outputs[{{}, i}], targets[{{}, i+1}])




seq_length = 10
batch_size = 1
n_neurons = 20

text = batcher.load_text()
alphabet, train_batcher = batcher.make_batch_iterators(text, torch.Tensor{1}, 10, 1)
batch = train_batcher[1]()
initial_state = torch.zeros(1, n_neurons)

modules = gru.build(seq_length, table.size(alphabet), n_neurons)
outputs, loss = forward(modules, batch, initial_state)
print(loss)
print(encoding.one_hot_to_chars(alphabet, outputs[1]))
