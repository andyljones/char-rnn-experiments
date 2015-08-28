local torch = require 'torch'
local networktools = require 'networktools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons)
  local hidden = nn.Tanh()(networktools.compose_inputs(input_size, n_neurons, input, prev_hidden, ''))
  local output = nn.LogSoftMax()(nn.Linear(n_neurons, input_size)(hidden):annotate{name='out'})

  return hidden, output
end

function M.build(n_timesteps, n_symbols, n_neurons)
  local input = nn.Identity()()
  local inputs = {nn.SplitTable(1, 2)(input):split(n_timesteps)}

  local outputs = {}

  local initial_state = nn.Identity()()
  local hidden_state = initial_state
  for i = 1, n_timesteps do
    local new_hidden_state, output = M.build_cell(inputs[i], hidden_state, n_symbols, n_neurons)
    outputs[i] = nn.Reshape(-1, 1, n_symbols, false)(output)
    hidden_state = new_hidden_state
  end

  local output = nn.JoinTable(2)(outputs)

  module = nn.gModule({input, initial_state}, {output, hidden_state})
  -- networktools.share_matched_names(module)
  return module
end

return M
