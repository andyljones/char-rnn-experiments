local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
local initializer = require 'initializer'
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

      loss = loss + timestep_loss/n_timesteps_minus_1
      grad_loss[{{}, i}] = timestep_grad_loss/n_timesteps_minus_1
    end

    return loss, grad_loss
end

function make_feval(model, training_iterator, n_neurons, grad_clip)
  function feval(x)
    local params, grad_params = model:getParameters()
    params:copy(x)
    grad_params:zero()

    local X, y = training_iterator()

    local zero_state = torch.zeros(X:size(1), n_neurons)

    local output, _ = unpack(model:forward({X, zero_state}))
    local loss, grad_loss = calculate_loss(output, y)
    model:backward({X, zero_state}, {grad_loss, zero_state})

    grad_params:clamp(-grad_clip, grad_clip)

    return loss, grad_params
  end

  return feval
end

function build_model(options)
  local text = batcher.load_text()
  local alphabet, batch_iterators = batcher.make_batch_iterators(
                                                                  text,
                                                                  torch.Tensor(options.split),
                                                                  options.n_timesteps,
                                                                  options.n_samples
                                                                )

  local model = gru.build(options.n_timesteps-1, table.size(alphabet), options.n_neurons)
  initializer.initialize_network(model)

  return model, alphabet, batch_iterators
end

function train(options)
  local model, alphabet, iterators = build_model(options)
  local feval = make_feval(model, iterators[1], options.n_neurons, options.grad_clip)
  local params, _ = model:getParameters()

  for i = 1, options.n_steps do
    local _, loss = optim.rmsprop(feval, params, options.optim_state)
    print(i, loss[1])
  end
end

options = {
  n_neurons = 128,
  n_timesteps = 50,
  n_samples = 50,
  optim_state = {learningRate=5e-3, alpha=0.95},
  split = {0.95, 0.05},
  grad_clip = 5,
  n_steps = 100
}

train(options)

return {
  build_model=build_model,
  initialize=initialize,
  calculate_loss=calculate_loss
}
