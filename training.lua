local batching = require 'batching'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
local initializer = require 'initializer'
local saving = require 'saving'
require 'nn'
require 'nngraph'
require 'optim'

local M = {}

function M.make_iterators(options)
  local text = batching.load_text()
  return batching.make_batch_iterators(text, torch.Tensor(options.split), options.n_timesteps, options.n_samples)
end

function M.make_model(options, n_symbols)
  local model = gru.build(options.n_samples, options.n_timesteps-1, n_symbols, options.n_neurons)
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

function M.make_trainer(model, training_iterator, grad_clip)
  function trainer(x)
    local params, grad_params = model:getParameters()
    params:copy(x)
    grad_params:zero()

    local X, y = training_iterator()

    local output, _ = unpack(model:forward({X, model.default_state}))
    local loss, grad_loss = M.calculate_loss(output, y)
    model:backward({X, model.default_state}, {grad_loss, model.default_state})

    grad_params:clamp(-grad_clip, grad_clip)

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

function M.train(model, iterators, saver)
  local trainer = M.make_trainer(model, iterators[1], options.grad_clip)
  local tester = M.make_tester(model, iterators[2], options.n_test_batches)
  local params, _ = model:getParameters()

  local train_losses, test_losses = {}, {}

  for i = 1, options.n_steps do
    local _, loss = optim.adam(trainer, params, options.optim_state)
    train_losses[i] = loss
    print(string.format('Batch %4d, loss %4.2f', i, loss[1]))

    if i % options.testing_interval == 0 then
      local loss = tester()
      test_losses[i] = loss
      print(string.format('Test loss %.2f', loss))

      if saver then
        print(string.format('Saving...'))
        saver(model, train_losses, test_losses)
      end
    end
  end
end

function M.run(options)
  local start_time = os.time()
  local alphabet, iterators = M.make_iterators(options)
  local model = M.make_model(options, table.size(alphabet))
  local saver = saving.make_saver(options, alphabet, start_time)
  M.train(model, iterators, saver)
end

options = {
  n_neurons = 128,
  n_timesteps = 50,
  n_samples = 50,
  optim_state = {learningRate=5e-3, alpha=0.95},
  split = {0.95, 0.05},
  grad_clip = 5,
  n_steps = 1000,
  n_test_batches = 10,
  testing_interval = 100,
}

M.run(options)

return M
