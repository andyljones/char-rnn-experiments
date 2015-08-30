local torch = require 'torch'
local buildtools = require 'buildtools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons)
  local reset_gate = nn.Sigmoid()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, 'reset'))
  local reset_hidden = nn.CMulTable()({reset_gate, prev_hidden})
  local transformed_reset_hidden = nn.Linear(n_neurons, n_neurons)(reset_hidden):annotate{name='trh'}
  local transformed_input = nn.Linear(input_size, n_neurons)(input):annotate{name='ti'}
  local candidate_hidden = nn.Tanh()(nn.CAddTable()({transformed_input, transformed_reset_hidden}))

  local update_gate = nn.Sigmoid()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, 'update'))
  local update_candidate = nn.CMulTable()({update_gate, candidate_hidden})
  local update_compliment = nn.AddConstant(1)(nn.MulConstant(-1)(update_gate))
  local update_compliment_prev = nn.CMulTable()({update_compliment, prev_hidden})
  local next_hidden = nn.CAddTable()({update_candidate, update_compliment_prev})

  return next_hidden
end

function M.build(n_symbols, n_neurons)
  local input = nn.Identity()()
  local prev_hidden = nn.Identity()()

  local next_hidden = M.build_cell(input, prev_hidden, n_symbols, n_neurons)
  local output = nn.Linear(n_neurons, n_symbols)(next_hidden):annotate{name='out'}

  local module = nn.gModule({input, prev_hidden}, {output, next_hidden})

  module.config = {}
  module.config.n_neurons = n_neurons
  module.config.n_symbols = n_symbols

  return module
end

return M
