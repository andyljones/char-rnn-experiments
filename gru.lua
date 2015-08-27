local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
require 'graph'

function compose_inputs(input_size, n_neurons, input, prev_hidden)
  local input_to_hidden = nn.Linear(input_size, n_neurons)(input)
  local hidden_to_hidden = nn.Linear(n_neurons, n_neurons)(prev_hidden)
  return nn.CAddTable()({input_to_hidden, hidden_to_hidden})
end

function build_cell(input, prev_hidden, input_size, n_neurons)
  local reset_gate = nn.Sigmoid()(compose_inputs(input_size, n_neurons, input, prev_hidden))
  local reset_hidden = nn.CMulTable()({reset_gate, prev_hidden})
  local transformed_reset_hidden = nn.Linear(n_neurons, n_neurons)(reset_hidden)
  local transformed_input = nn.Linear(input_size, n_neurons)(input)
  local candidate_hidden = nn.Tanh()(nn.CAddTable()({transformed_input, transformed_reset_hidden}))

  local update_gate = nn.Sigmoid()(compose_inputs(input_size, n_neurons, input, prev_hidden))
  local update_candidate = nn.CMulTable()({update_gate, candidate_hidden})
  local update_compliment = nn.AddConstant(1)(nn.MulConstant(-1)(update_gate))
  local update_compliment_prev = nn.CMulTable()({update_compliment, prev_hidden})
  local next_hidden = nn.CAddTable()({update_candidate, update_compliment_prev})

  return next_hidden
end

function build(n_timesteps, n_symbols, n_neurons)
  local input = nn.Identity()()
  local inputs = {input:split(n_timesteps)}

  local modules = {}
  local outputs = {}

  local initial_state = nn.Identity()()
  local hidden_state = initial_state
  for i = 1, n_timesteps do
    local new_hidden_state = build_cell(inputs[i], hidden_state, n_symbols, n_neurons)
    local output = nn.LogSoftMax()(nn.Linear(n_neurons, n_symbols)(new_hidden_state))
    outputs[i] = nn.Reshape(-1, 1, n_symbols, false)(output)
    -- modules[i] = nn.gModule({inputs[i], hidden_state}, {outputs[i], new_hidden_state})
    hidden_state = new_hidden_state
  end

  local output = nn.JoinTable(2)(outputs)

  return nn.gModule({input, initial_state}, {output, hidden_state})
end

return {build=build}
