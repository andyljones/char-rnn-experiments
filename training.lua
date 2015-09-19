local mirrorbatching = require 'mirrorbatching'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'
local initializer = require 'initializer'
local storage = require 'storage'
local usetools = require 'usetools'
local timing = require 'timing'
local profi = require 'profi'
local rnn = require 'rnn'

require 'optim'
require 'nn'
require 'nngraph'

local M = {}

function M.make_iterators(options)
  return mirrorbatching.make_batch_iterators(options.n_timesteps, options.n_samples)
end

function M.make_model(options, n_symbols)
  local model = rnn.build(n_symbols, options.n_neurons, options.n_layers):cuda()
  model.params, model.param_grads = model:getParameters()
  initializer.initialize_network(model)
  return model
end

function M.make_loss_calculator()
  local criterion = nn.CrossEntropyCriterion():cuda()

  function co(output, y)
    local n_samples, n_timesteps, n_symbols = unpack(torch.totable(output:size()))
    local loss = 0
    local grad_loss = torch.zeros(n_samples, n_timesteps, n_symbols):cuda()
    for i = 1, n_timesteps do
      local timestep_loss = criterion:forward(output[{{}, i}], y[{{}, i}])
      local timestep_grad_loss = criterion:backward(output[{{}, i}], y[{{}, i}])

      loss = loss + timestep_loss/n_timesteps
      grad_loss[{{}, i}] = timestep_grad_loss/n_timesteps
    end

    assert(loss == loss)

    return loss, grad_loss
  end

  return co
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
  local calculate_loss = M.make_loss_calculator()

  function trainer(x)
    model.params:copy(x)
    local X, y = training_iterator()
    local X, y = X:float():cuda(), y:float():cuda()

    local output = forward(X)
    local loss, grad_loss = calculate_loss(output, y)
    backward(grad_loss)

    clip(model.param_grads, grad_clip)

    return loss, model.param_grads
  end

  return trainer
end

function M.make_tester(model, testing_iterator, n_test_batches)
  local n_timesteps = testing_iterator():size(2)
  local forward, _ = usetools.make_forward_backward(model, n_timesteps)
  local calculate_loss = M.make_loss_calculator()

  function tester()
    local average_loss = 0
    for i = 1, n_test_batches do
      local X, y = testing_iterator()
      local X, y = X:cuda(), y:cuda()
      local output = forward(X)
      local loss, _  = calculate_loss(output, y)
      average_loss = average_loss + loss/n_test_batches
    end
    return average_loss
  end

  return tester
end

function M.stop_early(test_losses)
  local test_losses = torch.Tensor(test_losses)
  local n_losses = test_losses:size(1)
  if n_losses > 1 then
    local previous_min = test_losses[{{1, n_losses - 1}}]:min()
    local recent_min = test_losses[n_losses]
    return (recent_min > 0.99*previous_min) or (0.01 > recent_min)
  else
    return false
  end
end

function M.train(model, iterators, options, saver)
  local trainer = M.make_trainer(model, iterators[1], options.grad_clip)
  local tester = M.make_tester(model, iterators[2], options.n_test_batches)
  local timer = timing.make_timer()

  local train_losses, test_losses = {}, {}

  for i = 1, options.max_steps do
    local _, loss = optim.rmsprop(trainer, model.params, options.optim_state)
    train_losses[i] = loss

    local time_per_batch = 1000*timer()
    if i%100 == 1 then
      print(string.format('Batch %4d, loss %5.3f, %.0fms per batch', i, loss[1], time_per_batch))
    end

    if i % options.testing_interval == 0 then
      local loss = tester()
      test_losses[#test_losses+1] = loss
      print(string.format('Test loss %.3f', loss))

      if saver then
        print(string.format('Saving...'))
        saver(model.params, train_losses, test_losses)
      end

      if M.stop_early(test_losses) then break end
    end

    if i % 10 == 0 then
      collectgarbage()
    end
  end
end


local cuda_initialized = false
function M.initialize_cuda()
  if not cuda_initialized then
    local cunn = require 'cunn'
    local cutorch = require 'cutorch'
    cutorch.setDevice(1)
  end
end

function M.run(options)
  M.initialize_cuda()
  local start_time = os.time()
  local alphabet, iterators = M.make_iterators(options)
  local model = M.make_model(options, #alphabet)
  print(string.format('This model has %d parameters', model.params:size(1)))

  local saver = storage.make_saver(model, options, alphabet, start_time)
  M.train(model, iterators, options, saver)
end

return M
