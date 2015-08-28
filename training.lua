local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
require 'optim'

function decode(alphabet, batch)
  local results = {}
  for i = 1, batch:size(1) do
    results[i] = encoding.one_hot_to_chars(alphabet, batch[i])
  end

  return results
end

function batch_one_hot_to_ints(batch)
  return select(2, batch:max(3)):view(batch:size(1), batch:size(2))
end

function calculate_loss(output, input)
    local n_samples, n_timesteps, n_symbols = unpack(torch.totable(input:size()))

    local relevant_outputs = output[{{}, {1, -2}}]:clone():view(-1, n_symbols)
    local targets = batch_one_hot_to_ints(input)[{{}, {2, -1}}]:clone():view(-1)

    local criterion = nn.ClassNLLCriterion()
    local loss = criterion:forward(relevant_outputs, targets)
    local grad_loss = criterion:backward(relevant_outputs, targets):view(n_samples, n_timesteps - 1, n_symbols)
    local grad_loss = torch.cat(grad_loss, torch.zeros(n_samples, 1, n_symbols), 2)

    return loss, grad_loss
end

function make_feval(model, training_iterator, n_neurons, grad_clip)
  function feval(_)
    local params, grad_params = model:getParameters()
    grad_params:zero()

    local input = training_iterator()

    local n_samples = input:size(1)
    local initial_state = torch.zeros(n_samples, n_neurons)
    local final_state = torch.zeros(n_samples, n_neurons)

    local output, final_state = unpack(model:forward({input, initial_state}))
    local loss, grad_loss = calculate_loss(output, input)
    model:backward({input, initial_state}, {grad_loss, final_state})

    grad_params:clamp(-grad_clip, grad_clip)

    return loss, grad_params
  end

  return feval
end

function initialize(model)
  local params, _ = model:getParameters()
  params:uniform(-0.08, 0.08)
end

local n_neurons, n_timesteps, n_samples = 128, 50, 50
local grad_clip = 5
local optim_state = {learningRate = 2e-3, alpha=0.95}

local text = batcher.load_text()
local alphabet, batch_iterators = batcher.make_batch_iterators(text, torch.Tensor{1}, n_timesteps, n_samples)

local training_iterator = batch_iterators[1]
local model = gru.build(n_timesteps, table.size(alphabet), n_neurons)
initialize(model)

local feval = make_feval(model, training_iterator, n_neurons, grad_clip)
local params, _ = model:getParameters()
for i = 1, 100 do
  local _, loss = optim.rmsprop(feval, params, optim_state)
  print(i, loss[1])
end
