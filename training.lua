local batching = require 'batching'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
local initializer = require 'initializer'
local storage = require 'storage'
require 'nn'
require 'nngraph'
require 'optim'

local M = {}

function M.make_iterators(options)
  local text = batching.load_text()
  return batching.make_batch_iterators(text, torch.Tensor(options.split), options.n_timesteps, options.n_samples)
end

function M.make_model(options, n_symbols)
  local model = gru.build(n_symbols, options.n_neurons)
  model.params, model.param_grads = model:getParameters()
  initializer.initialize_network(model)
  return model
end

function M.calculate_loss(output, y)
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

function clip(gradients, threshold)
  local magnitude = torch.norm(gradients)
  if magnitude > threshold then
    gradients:copy(gradients:mul(threshold/magnitude))
  end
end

function M.make_trainer(model, training_iterator, grad_clip)
  function trainer(x)
    model.params:copy(x)
    model.grad_params:zero()

    local X, y = training_iterator()

    local output, _ = unpack(model:forward({X, model.default_state}))
    local loss, grad_loss = M.calculate_loss(output, y)
    model:backward({X, model.default_state}, {grad_loss, model.default_state})

    clip(grad_params, grad_clip)

    return loss, grad_params
  end

  return trainer
end

function M.make_tester(model, testing_iterator, n_test_batches)
  function tester()
    local average_loss = 0
    for i = 1, n_test_batches do
      local X, y = testing_iterator()
      local output, _ = unpack(model:forward({X, model.default_state}))
      local loss, _  = M.calculate_loss(output, y)
      average_loss = average_loss + loss/n_test_batches
    end
    return average_loss
  end

  return tester
end

function M.adjust_lr(test_losses, optim_state)
  local test_iterations = table.sort(table.keys(test_losses))
  local compacted_losses = torch.Tensor(#test_iterations)
  for i = 1, #test_iterations do
    compacted_losses[i] = test_losses[test_iterations[i]]
  end

  if compacted_losses:size(1) > 5 then
    local historical_best = compacted_losses[{{1, -6}}]:min()
    local recent_best = compacted_losses[{{-5, -1}}]:min()
    if recent_best > historical_best - 0.01 then
      optim_state.learningRate = 0.1*optim_state.learningRate
      print(string.format('Learning rate updated to %f', optim_state.learningRate))
    end
  end
end

function M.train(model, iterators, saver)
  local trainer = M.make_trainer(model, iterators[1], options.grad_clip)
  local tester = M.make_tester(model, iterators[2], options.n_test_batches)

  local train_losses, test_losses = {}, {}

  for i = 1, options.n_steps do
    local _, loss = optim.adam(trainer, model.params, options.optim_state)
    train_losses[i] = loss
    print(string.format('Batch %4d, loss %4.2f', i, loss[1]))

    if i % options.testing_interval == 0 then
      local loss = tester()
      test_losses[i] = loss
      print(string.format('Test loss %.2f', loss))

      M.adjust_lr(test_losses, options.optim_state)

      if saver then
        print(string.format('Saving...'))
        saver(params, train_losses, test_losses)
      end
    end

    if i % 10 == 0 then
      collectgarbage()
    end
  end
end

function M.run(options)
  local start_time = os.time()
  local alphabet, iterators = M.make_iterators(options)
  local model = M.make_model(options, table.size(alphabet))
  local saver = storage.make_saver(model, options, alphabet, start_time)
  M.train(model, iterators, saver)
end

options = {
  n_neurons = 256,
  n_timesteps = 50,
  n_samples = 50,
  optim_state = {learningRate=5e-3, alpha=0.95},
  split = {0.95, 0.05},
  grad_clip = 0.5,
  n_steps = 10000,
  n_test_batches = 10,
  testing_interval = 100,
}

-- M.run(options)

return M
