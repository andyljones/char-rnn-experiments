local batching = require 'batching'
local gru = require 'gru'
local rnn = require 'rnn'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
local initializer = require 'initializer'
local storage = require 'storage'
local usetools = require 'usetools'
require 'nn'
require 'nngraph'
require 'optim'

local M = {}

function M.make_iterators(options)
  local text = batching.load_text()
  return batching.make_batch_iterators(text, torch.Tensor(options.split), options.n_timesteps, options.n_samples)
end

function M.make_model(options, n_symbols)
  local model = rnn.build(n_symbols, options.n_neurons, options.n_layers)
  model.params, model.param_grads = model:getParameters()
  initializer.initialize_network(model)
  return model
end

function M.calculate_loss(output, y)
  local n_samples, n_timesteps, n_symbols = unpack(torch.totable(output:size()))
  local loss = 0
  local grad_loss = torch.zeros(n_samples, n_timesteps, n_symbols)
  for i = 1, n_timesteps do
    local criterion = nn.CrossEntropyCriterion()
    local timestep_loss = criterion:forward(output[{{}, i}], y[{{}, i}])
    local timestep_grad_loss = criterion:backward(output[{{}, i}], y[{{}, i}])

    loss = loss + timestep_loss/n_timesteps
    grad_loss[{{}, i}] = timestep_grad_loss/n_timesteps
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
  local n_timesteps = training_iterator():size(2)
  local forward, backward = usetools.make_forward_backward(model, n_timesteps)
  function trainer(x)
    model.params:copy(x)
    local X, y = training_iterator()

    local output = forward(X)
    local loss, grad_loss = M.calculate_loss(output, y)
    backward(grad_loss)

    clip(model.param_grads, grad_clip)

    return loss, model.param_grads
  end

  return trainer
end

function M.make_tester(model, testing_iterator, n_test_batches)
  local n_timesteps = testing_iterator():size(2)
  local forward, _ = usetools.make_forward_backward(model, n_timesteps)
  function tester()
    local average_loss = 0
    for i = 1, n_test_batches do
      local X, y = testing_iterator()
      local output = forward(X)
      local loss, _  = M.calculate_loss(output, y)
      average_loss = average_loss + loss/n_test_batches
    end
    return average_loss
  end

  return tester
end

function M.train(model, iterators, saver, options)
  local trainer = M.make_trainer(model, iterators[1], options.grad_clip)
  local tester = M.make_tester(model, iterators[2], options.n_test_batches)

  local train_losses, test_losses = {}, {}

  for i = 1, options.n_steps do
    local _, loss = optim.rmsprop(trainer, model.params, options.optim_state)
    train_losses[i] = loss
    print(string.format('Batch %4d, loss %4.2f', i, loss[1]))

    if i % options.testing_interval == 0 then
      local loss = tester()
      test_losses[i] = loss
      print(string.format('Test loss %.2f', loss))

      if saver then
        print(string.format('Saving...'))
        saver(model.params, train_losses, test_losses)
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
  local model = M.make_model(options, #alphabet)
  print(string.format('This model has %d parameters', model.params:size(1)))

  local saver = storage.make_saver(model, options, alphabet, start_time)
  M.train(model, iterators, saver, options)
end

options = {
  n_layers = 1,
  n_neurons = 128,
  n_timesteps = 50,
  n_samples = 50,
  optim_state = {learningRate=1e-3, alpha=0.95},
  split = {0.95, 0.05},
  grad_clip = 5,
  n_steps = 10000,
  n_test_batches = 100,
  testing_interval = 1000,
}

M.run(options)

return M
