local torch = require 'torch'
local nn = require 'nn'
local nngraph = require 'nngraph'

function compose_inputs(input_size, n_neurons, input, prev_hidden)
  local input_to_hidden = nn.Linear(input_size, n_neurons)(input)
  local hidden_to_hidden = nn.Linear(n_neurons, n_neurons)(prev_hidden)
  return nn.CAddTable()({input_to_hidden, hidden_to_hidden})
end

function build_layer(input_size, n_neurons)
  local input = nn.Identity()()
  local prev_hidden = nn.Identity()()

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

  nngraph.annotateNodes()

  return input, prev_hidden, next_hidden
end

function build(n_symbols, n_neurons, n_layers)
end
