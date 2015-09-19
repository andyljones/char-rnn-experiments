local table = require 'std.table'
local training = require 'training'

function make_param_generator(timestep_stride, size_stride)
  local timesteps = torch.linspace(timestep_stride[1], timestep_stride[2], timestep_stride[3])
  local sizes = torch.linspace(size_stride[1], size_stride[2], size_stride[3])

  function co()
    local count = 1
    for k = 1, timesteps:size(1) + sizes:size(1) - 1 do
      for j = k, 1, -1 do
        local i = k - j + 1
        if i <= timesteps:size(1) and j <= sizes:size(1) then
          local options = {
            n_layers = 1,
            n_neurons = math.floor(sizes[j]),
            n_timesteps = 2*math.floor(timesteps[i]/2)+1,
            n_samples = 100,
            optim_state = {learningRate=1e-3, alpha=0.95},
            grad_clip = 5,
            max_steps = 25000,
            n_test_batches = 100,
            testing_interval = 1000,
            name = 'saturation-1',
            save_dir = '/mnt/saturation-records'
          }

          coroutine.yield(count, options)
          count = count + 1
        end
      end
    end
  end

  return coroutine.wrap(co)
end

local param_gen = make_param_generator({3, 50, 11}, {1, 100, 11})
for i = 1, math.huge do
  local count, options = param_gen()
  if not options then break end
  print(string.format('Options: %d neurons, %d timesteps. Iteration %d', options.n_neurons, options.n_timesteps, count))
  training.run(options)
end
