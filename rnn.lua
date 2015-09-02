local torch = require 'torch'
local buildtools = require 'buildtools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons, layer)
  local hidden_1 = nn.ReLU()(buildtools.compose_inputs(input_size, n_neurons, input, prev_hidden, '1'))
  local hidden_2 = nn.ReLU()(buildtools.compose_inputs(input_size, n_neurons, input, hidden_1, '2'))

  return hidden_2
end

function M.build(n_symbols, n_neurons, n_layers)
  local input = nn.Identity()()
  local prev_hidden = nn.Identity()()
  local prev_hiddens

  if n_layers > 1 then
    prev_hiddens = {nn.SplitTable(1, 2)(prev_hidden):split(n_layers)}
  else
    prev_hiddens = {nn.Reshape(n_neurons, true)(prev_hidden)}
  end

  local next_hiddens = {M.build_cell(input, prev_hiddens[1], n_symbols, n_neurons, tostring(1))}
  for i = 2, n_layers do
    next_hiddens[i] = M.build_cell(next_hiddens[i-1], prev_hiddens[i], n_neurons, n_neurons, tostring(i))
  end

  local output = nn.Linear(n_neurons, n_symbols)(next_hiddens[n_layers]):annotate{name='out'}

  for i = 1, n_layers do next_hiddens[i] = nn.Reshape(1, n_neurons, true)(next_hiddens[i]) end

  local next_hidden
  if n_layers > 1 then
    next_hidden = nn.JoinTable(1, 2)(next_hiddens)
  else
    next_hidden = next_hiddens[1]
  end

  local module = nn.gModule({input, prev_hidden}, {output, next_hidden})

  module.config = {}
  module.config.n_layers = n_layers
  module.config.n_neurons = n_neurons
  module.config.n_symbols = n_symbols

  return module
end

return M
