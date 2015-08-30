local torch = require 'torch'
local networktools = require 'networktools'
require 'nn'
require 'nngraph'

local M = {}

function M.build_cell(input, prev_hidden, input_size, n_neurons)
  local reset_gate = nn.Sigmoid()(networktools.compose_inputs(input_size, n_neurons, input, prev_hidden, 'reset'))
  local reset_hidden = nn.CMulTable()({reset_gate, prev_hidden})
  local transformed_reset_hidden = nn.Linear(n_neurons, n_neurons)(reset_hidden):annotate{name='trh'}
  local transformed_input = nn.Linear(input_size, n_neurons)(input):annotate{name='ti'}
  local candidate_hidden = nn.Tanh()(nn.CAddTable()({transformed_input, transformed_reset_hidden}))

  local update_gate = nn.Sigmoid()(networktools.compose_inputs(input_size, n_neurons, input, prev_hidden, 'update'))
  local update_candidate = nn.CMulTable()({update_gate, candidate_hidden})
  local update_compliment = nn.AddConstant(1)(nn.MulConstant(-1)(update_gate))
  local update_compliment_prev = nn.CMulTable()({update_compliment, prev_hidden})
  local next_hidden = nn.CAddTable()({update_candidate, update_compliment_prev})

  local output = nn.LogSoftMax()(nn.Linear(n_neurons, input_size)(next_hidden):annotate{name='out'})

  return next_hidden, output
end

function M.build(n_samples, n_timesteps, n_symbols, n_neurons)
  local input = nn.Identity()()
  local inputs = {nn.SplitTable(1, 2)(input):split(n_timesteps)}

  local outputs = {}

  local initial_state = nn.Identity()()
  local hidden_state = initial_state
  for i = 1, n_timesteps do
    local new_hidden_state, output = M.build_cell(inputs[i], hidden_state, n_symbols, n_neurons)
    new_hidden_state:annotate{timestep=string.format('hidden-%d', i)}
    outputs[i] = nn.Reshape(-1, 1, n_symbols, false)(output)
    hidden_state = new_hidden_state
  end

  local output = nn.JoinTable(2)(outputs)

  module = nn.gModule({input, initial_state}, {output, hidden_state})
  networktools.share_matched_names(module)

  module.default_state = torch.zeros(n_samples, n_neurons)

  return module
end

return M
