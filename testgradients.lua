local luaunit = require 'luaunit'
local torch = require 'torch'
local batcher = require 'batcher'
local gru = require 'gru'
local table = require 'std.table'
local training = require 'training'
require 'nn'

function numerical_gradient(model, X, y, n_neurons, magnitude, param_index)
  local zero_state = torch.zeros(X:size(1), n_neurons)
  local params, _ = model:getParameters()
  local original_value = params[param_index]

  params[param_index] = original_value + magnitude
  local first_output, _ = unpack(model:forward({X, zero_state}))
  local first_loss, _ = training.calculate_loss(first_output, y)

  params[param_index] = original_value - magnitude
  local second_output, _ = unpack(model:forward({X, zero_state}))
  local second_loss, _ = training.calculate_loss(second_output, y)

  local gradient = (first_loss - second_loss)/(2*magnitude)

  params[param_index] = original_value

  return gradient
end

function analytic_gradients(model, X, y, n_neurons, magnitude)
  local zero_state = torch.zeros(X:size(1), n_neurons)
  local _, grad_params = model:getParameters()
  grad_params:zero()

  local output, final_state = unpack(model:forward({X, zero_state}))
  local _, grad_loss = calculate_loss(output, y)
  model:backward({x, initial_state}, {grad_loss, zero_state})

  return grad_params
end

function gradient_errors(n_checks, magnitude)
  local n_samples, n_timesteps, n_neurons = 50, 50, 128
  local model, iterators = training.build_model{n_timesteps=n_timesteps, n_neurons=n_neurons, n_samples=n_samples}
  local X, y = iterators[1]()

  local analytic_gradients = analytic_gradients(model, X, y, n_neurons, magnitude)

  local errors = torch.Tensor(n_checks)
  local n_params = select(1, model:getParameters()):size(1)
  for i = 1, n_checks do
    local index = torch.uniform(1, n_params)
    local numerical_gradient = numerical_gradient(model, X, y, n_neurons, magnitude, index)
    local analytic_gradient = analytic_gradients[index]
    errors[i] = math.abs(numerical_gradient - analytic_gradient)/analytic_gradient
    -- print(string.format('Numerical: %f; Analytic: %5f; Error: %f', numerical_gradient, analytic_gradient, relative_error))
  end

  return errors
end

function test_gradients()
  local errors = gradient_errors(10, 1e-4)
  luaunit.assertTrue(errors:max() < 1e-4)
end

luaunit.LuaUnit.run()
