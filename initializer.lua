require 'nn'
require 'nngraph'
local torch = require 'torch'
-- local training = require 'training'

local M = {}

function M.orthogonal_init(mat)
  assert(mat:size(1) == mat:size(2))
  local initScale = 1.
  local M = torch.randn(mat:size(1), mat:size(1))
  local Q, R = torch.qr(M)
  local Q = Q:mul(initScale)

  mat:copy(Q)
end

function M.glorot_init(mat, relu)
  local relu = relu or false
  local n_in, n_out = mat:size(1), mat:size(2)
  local limit = math.sqrt(6/(n_in + n_out))

  if relu then
    limit = math.sqrt(2)*limit
  end

  mat:uniform(-limit, limit)
end

function M.initialize_weights(module)
  local weights = module.weight
  if weights:size(1) == weights:size(2) then
    M.orthogonal_init(weights)
  else
    M.glorot_init(weights)
  end
  -- module.weight:uniform(-0.08, 0.08)
end

function M.initialize_biases(module)
  module.bias:zero()
end

function M.initialize_network(model)
  local nodes = model.fg.nodes
  local visited = {}
  local h2hs = {}
  for i = 1, #nodes do
    local node = nodes[i]
    local module = node.data.module
    local name = node.data.annotations.name
    if name and not visited[name] then
      if name == '1-h2h' then
        h2hs[1] = module.weight
        print('1 initialized')
      elseif name == '2-h2h' then
        h2hs[2] = module.weight
        print('2 initialized')
      else
        M.initialize_weights(module)
      end
      M.initialize_biases(module)
    end
  end

  M.orthogonal_init(h2hs[1])
  h2hs[2]:copy(h2hs[1]:t())

  model.param_grads:zero()
end

return M

-- local model = training.build_model{n_neurons=20, n_timesteps=10, n_samples=5, split={1.}}
-- initialize_network(model)
