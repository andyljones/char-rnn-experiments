local torch = require 'torch'
local nn = require 'nn'
require 'nngraph'
require 'graph'

function compose_inputs(input_size, n_neurons, input, prev_hidden)
  local input_to_hidden = nn.Linear(input_size, n_neurons)(input)
  local hidden_to_hidden = nn.Linear(n_neurons, n_neurons)(prev_hidden)
  return nn.CAddTable()({input_to_hidden, hidden_to_hidden})
end

function build_layer(input, input_size, n_neurons)
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

  return prev_hidden, next_hidden
end

function build(n_symbols, n_neurons)
  local input = nn.Identity()()
  local prev_hidden, next_hidden = build_layer(input, n_symbols, n_neurons)

  local gmod = nn.gModule({input, prev_hidden}, {next_hidden})
  return gmod
end

gmod = build(10, 20)
print(gmod)
graph.dot(gmod.fg, 'tmp', 'tmp')

-- return {build=build}
