local torch = require 'torch'
local buildtools = require 'buildtools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons)
  return nn.Tanh()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, ''))
end

function M.build(n_symbols, n_neurons)
  local input = nn.Identity()()
  local prev_hidden = nn.Identity()()

  local next_hidden = M.build_cell(input, prev_hidden, n_symbols, n_neurons)
  local output = nn.Linear(n_neurons, n_symbols)(next_hidden)

  local module = nn.gModule({input, prev_hidden}, {output, next_hidden})

  module.config = {}
  module.config.n_neurons = n_neurons
  module.config.n_symbols = n_symbols

  return module
end

return M
