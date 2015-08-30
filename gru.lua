local torch = require 'torch'
local buildtools = require 'buildtools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons, layer)
  local reset_gate = nn.Sigmoid()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, layer .. '-reset'))
  local reset_hidden = nn.CMulTable()({reset_gate, prev_hidden})
  local transformed_reset_hidden = nn.Linear(n_neurons, n_neurons)(reset_hidden):annotate{name=layer .. '-trh'}
  local transformed_input = nn.Linear(input_size, n_neurons)(input):annotate{name=layer .. '-ti'}
  local candidate_hidden = nn.Tanh()(nn.CAddTable()({transformed_input, transformed_reset_hidden}))

  local update_gate = nn.Sigmoid()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, layer .. '-update'))
  local update_candidate = nn.CMulTable()({update_gate, candidate_hidden})
  local update_compliment = nn.AddConstant(1)(nn.MulConstant(-1)(update_gate))
  local update_compliment_prev = nn.CMulTable()({update_compliment, prev_hidden})
  local next_hidden = nn.CAddTable()({update_candidate, update_compliment_prev})

  return next_hidden
end

function M.build(n_symbols, n_neurons, n_layers)
  local input = nn.Identity()()
  local prev_hidden = nn.Identity()()
  local prev_hiddens = {nn.SplitTable(1, 2)(prev_hidden):split(n_layers)}

  prev_hiddens[1] = nn.Reshape(n_neurons, true)(prev_hiddens[1])
  local next_hiddens = {M.build_cell(input, prev_hiddens[1], n_symbols, n_neurons, tostring(1))}
  for i = 2, n_layers do
    prev_hiddens[i] = nn.Reshape(n_neurons, true)(prev_hiddens[i])
    next_hiddens[i] = M.build_cell(next_hiddens[i-1], prev_hiddens[i], n_neurons, n_neurons, tostring(i))
  end

  local output = nn.Linear(n_neurons, n_symbols)(next_hiddens[n_layers]):annotate{name='out'}

  for i = 1, n_layers do
    next_hiddens[i] = nn.Reshape(1, n_neurons, true)(next_hiddens[i])
  end
  local next_hidden = nn.JoinTable(1, 2)(next_hiddens)


  local module = nn.gModule({input, prev_hidden}, {output, next_hidden})

  module.config = {}
  module.config.n_layers = n_layers
  module.config.n_neurons = n_neurons
  module.config.n_symbols = n_symbols

  return module
end

return M
