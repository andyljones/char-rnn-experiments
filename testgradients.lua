local luaunit = require 'luaunit'
local torch = require 'torch'
local batching = require 'batching'
local gru = require 'gru'
local table = require 'std.table'
local training = require 'training'
local usetools = require 'usetools'
local encoding = require 'encoding'
require 'nn'

function numerical_gradient(params, forward, X, y, magnitude, param_index)
  local original_value = params[param_index]

  params[param_index] = original_value + magnitude
  local first_outputs, _, _ = forward(X)
  local first_loss, _ = training.calculate_loss(first_outputs, y)

  params[param_index] = original_value - magnitude
  local second_outputs, _ = forward(X)
  local second_loss, _ = training.calculate_loss(second_outputs, y)

  local gradient = (first_loss - second_loss)/(2*magnitude)

  params[param_index] = original_value

  return gradient
end

function analytic_gradients(forward, backward, X, y)
  local outputs, modules, states = forward(X)
  local _, loss_grads = training.calculate_loss(outputs, y)
  local param_grads = backward(loss_grads)

  return param_grads
end

function make_input(n_samples, n_timesteps, n_symbols)
  local X = torch.Tensor(n_samples, n_timesteps, n_symbols)
  local y = torch.Tensor(n_samples, n_timesteps)
  for i = 1, n_samples do
    y[i]:uniform(1, n_symbols):round()
    X[i] = encoding.ints_to_one_hot(y[i], n_symbols)
  end

  return X, y
end

function gradient_errors(n_checks, magnitude)
  local options = {n_samples=50, n_timesteps=50, n_neurons=128, split={1.}}
  local n_symbols = 2
  local model = training.make_model(options, n_symbols)
  local X, y = make_input(options.n_samples, options.n_timesteps, n_symbols)
  local forward, backward = usetools.make_forward_backward(model, X:size(2))

  local analytic_gradients = analytic_gradients(forward, backward, X, y)

  local n_params = model.params:size(1)
  local errors = torch.Tensor(n_checks)
  for i = 1, n_checks do
    local index = torch.uniform(1, n_params)
    local numerical_gradient = numerical_gradient(model.params, forward, X, y, magnitude, index)
    local analytic_gradient = analytic_gradients[index]
    local error = math.abs(numerical_gradient - analytic_gradient)/math.abs(analytic_gradient)
    if error ~= error then errors[i] = 0 else errors[i] = error end
    print(string.format('Numerical: %f; Analytic: %f; Error: %f', numerical_gradient, analytic_gradient, errors[i]))
  end

  return errors
end

function test_gradients()
  local errors = gradient_errors(10, 1e-2)
  luaunit.assertTrue(errors:max() < 1e-3)
end

luaunit.LuaUnit.run()
