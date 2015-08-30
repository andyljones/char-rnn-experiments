local torch = require 'torch'
local gru = require 'gru'

local M = {}

function M.make_forward_backward(module, n_timesteps)
  local modules = {}
  for i = 1, n_timesteps do
    modules[i] = module:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end

  local states = {}
  local last_inputs = {}

  function forward(inputs)
    last_inputs = inputs

    local n_samples, _, n_symbols = unpack(torch.totable(inputs:size()))

    states = {torch.zeros(n_samples, module.config.n_neurons)}
    local outputs = torch.Tensor(n_samples, n_timesteps, n_symbols)
    for i = 1, n_timesteps do
      outputs[{{}, i}], states[i+1] = unpack(modules[i]:forward({inputs[{{}, i}], states[i]}))
    end

    return outputs, modules, states
  end

  function backward(output_grads)
    local n_samples, _, n_symbols = unpack(torch.totable(last_inputs:size()))
    module.param_grads:zero()

    local state_grads = {[n_timesteps+1]=torch.zeros(n_samples, modules[1].config.n_neurons)}
    for i = n_timesteps, 1, -1 do
      _, state_grads[i] = unpack(modules[i]:backward({last_inputs[{{}, i}], states[i]}, {output_grads[{{}, i}], state_grads[i+1]}))
    end

    return module.param_grads
  end

  return forward, backward
end

return M
