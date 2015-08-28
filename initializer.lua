require 'nn'
require 'nngraph'
local torch = require 'torch'
-- local training = require 'training'

local M = {}

function M.initialize_weights(module)
  local weights = module.weight
  local n_in, n_out = weights:size(1), weights:size(2)
  local limit = math.sqrt(6/(n_in + n_out))
  weights:uniform(-limit, limit)
end

function M.initialize_biases(module)
  module.bias:zero()
end

function M.initialize_network(model)
  local nodes = model.fg.nodes
  local visited = {}
  for i = 1, #nodes do
    local node = nodes[i]
    local module = node.data.module
    local name = node.data.annotations.name
    if name and not visited[name] then
      M.initialize_weights(module)
      M.initialize_biases(module)
    end
  end
end

return M

-- local model = training.build_model{n_neurons=20, n_timesteps=10, n_samples=5, split={1.}}
-- initialize_network(model)
