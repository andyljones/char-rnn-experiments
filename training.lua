local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
require 'nn'
require 'nngraph'
require 'optim'

function decode(alphabet, batch)
  local results = {}
  for i = 1, batch:size(1) do
    results[i] = encoding.one_hot_to_chars(alphabet, batch[i])
  end

  return results
end

function calculate_loss(output, y)
    local n_samples, n_timesteps_minus_1, n_symbols = unpack(torch.totable(output:size()))

    local loss = 0
    local grad_loss = torch.zeros(n_samples, n_timesteps_minus_1, n_symbols)
    for i = 1, n_timesteps_minus_1 do
      local timestep_outputs = output[{{}, i}]
      local timestep_targets = y[{{}, i}]
      local criterion = nn.ClassNLLCriterion()
      local timestep_loss = criterion:forward(timestep_outputs, timestep_targets)
      local timestep_grad_loss = criterion:backward(timestep_outputs, timestep_targets)

      loss = loss + timestep_loss
      grad_loss[{{}, i}] = timestep_grad_loss/n_timesteps_minus_1
    end

    return loss/n_timesteps_minus_1, grad_loss
end

function make_feval(model, training_iterator, n_neurons, grad_clip)
  function feval(_)
    local params, grad_params = model:getParameters()
    grad_params:zero()

    local input = training_iterator()

    local zero_state = torch.zeros(input:size(1), n_neurons)

    local output, final_state = unpack(model:forward({input, zero_state}))
    local loss, grad_loss = calculate_loss(output, input)
    model:backward({input, zero_state}, {grad_loss, zero_state})

    grad_params:clamp(-grad_clip, grad_clip)

    return loss, grad_params
  end

  return feval
end

function initialize(model)
  local params, _ = model:getParameters()
  params:uniform(-0.08, 0.08)
end

function build_model(options)
  local n_neurons = options.n_neurons or 128
  local n_timesteps = options.n_timesteps or 50
  local n_samples = options.n_samples or 50

  local text = batcher.load_text()
  local alphabet, batch_iterators = batcher.make_batch_iterators(text, torch.Tensor{1}, n_timesteps, n_samples)

  local model = gru.build(n_timesteps - 1, table.size(alphabet), n_neurons)
  initialize(model)

  return model, batch_iterators
end
--
-- local grad_clip = options.grad_clip or 5
-- local optim_state = options.optim_state or {learningRate=2e-3, alpha=0.95}

-- local feval = make_feval(model, training_iterator, n_neurons, grad_clip)
-- local params, _ = model:getParameters()
--
-- for i = 1, 5 do
--   local _, loss = optim.rmsprop(feval, params, optim_state)
--   print(i, loss[1])
-- end

return {
  build_model=build_model,
  initialize=initialize,
  calculate_loss=calculate_loss
}
