

DEFAULT_OPTIONS = {
  n_layers = 1,
  n_neurons = nil,
  n_timesteps = nil,
  n_samples = 50,
  optim_state = {learningRate=1e-3, alpha=0.95},
  split = {0.95, 0.05},
  grad_clip = 5,
  n_steps = 10000,
  n_test_batches = 100,
  testing_interval = 1000
}

function make_param_generator(timestep_stride, size_stride)
  local timesteps = torch.linspace(timestep_stride[1], timestep_stride[2], timestep_stride[3])
  local sizes = torch.linspace(size_stride[1], size_stride[2], size_stride[3])

  function co()
    for k = 1, timesteps:size(1) + sizes:size(1) - 1 do
      for j = k, 1, -1 do
        local i = k - j + 1
        if i <= timesteps:size(1) and j <= sizes:size(1) then
          local options = DEFAULT_OPTIONS:clone()
          options.n_timesteps = timesteps[i]
          options.n_neurons = sizes[i]
          coroutine.yield(options)
        end
      end
    end
  end

  return coroutine.wrap(co)
end

make_param_generator({1, 3, 3}, {1, 5, 5})
