require 'nn'
require 'nngraph'

local M = {}

function M.compose_inputs(input_size, n_neurons, input, prev_hidden, name)
  local input_to_hidden = nn.Linear(input_size, n_neurons)(input):annotate{name=name..'-i2h'}
  local hidden_to_hidden = nn.Linear(n_neurons, n_neurons)(prev_hidden):annotate{name=name..'-h2h'}
  return nn.CAddTable()({input_to_hidden, hidden_to_hidden})
end

return M
