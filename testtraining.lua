local luaunit = require 'luaunit'
local torch = require 'torch'
local batcher = require 'batcher'
local gru = require 'gru'
local table = require 'std.table'
local training = require 'training'
require 'nn'

function numerical_gradient(model, input, initial_state, magnitude, param_index)
  local params, _ = model:getParameters()
  local original_value = params[param_index]

  params[param_index] = original_value + magnitude
  local first_output, _ = unpack(model:forward({input, initial_state}))
  local first_loss, _ = training.calculate_loss(first_output, input)

  params[param_index] = original_value - magnitude
  local second_output, _ = unpack(model:forward({input, initial_state}))
  local second_loss, _ = training.calculate_loss(second_output, input)

  local gradient = (second_loss - first_loss)/(2*magnitude)

  params[param_index] = original_value

  return gradient
end

function analytic_gradients(model, input, initial_state, magnitude)
  local _, grad_params = model:getParameters()
  grad_params:zero()

  local output, final_state = unpack(model:forward({input, initial_state}))
  local _, grad_loss = calculate_loss(output, input)
  model:backward({x, initial_state}, {grad_loss, final_state})

  return grad_params
end

function check_gradients()
  local n_samples, n_timesteps, n_neurons = 1, 50, 128
  local num_checks, magnitude = 10, 1e-5
  local model, iterators = training.build_model{n_samples=n_samples, n_timesteps=n_timesteps, n_neurons=n_neurons}
  local input = iterators[1]()
  local initial_state = torch.zeros(n_samples, n_neurons)

  local analytic_gradients = analytic_gradients(model, input, initial_state, magnitude)

  local n_params = select(1, model:getParameters()):size(1)
  for i = 1, num_checks do
    local index = torch.uniform(0, n_params)
    local numerical_gradient = numerical_gradient(model, input, initial_state, magnitude, index)
    local analytic_gradient = analytic_gradients[index]
    local relative_error = numerical_gradient/analytic_gradient
    print(string.format('Analytic: %5f; Numerical: %f. Error: %f', numerical_gradient, analytic_gradient, relative_error))
  end
end

check_gradients()
