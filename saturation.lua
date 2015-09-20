local table = require 'std.table'
local training = require 'training'
local lfs = require 'lfs'
local storage = require 'storage'
local torch = require 'torch'
require 'cutorch'

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

function run()
  local param_gen = make_param_generator({3, 50, 11}, {1, 100, 11})
  for i = 1, math.huge do
    local count, options = param_gen()
    if not options then break end
    print(string.format('Options: %d neurons, %d timesteps. Iteration %d', options.n_neurons, options.n_timesteps, count))
    training.run(options)
  end
end

function load_experiment_results(experiment_name)
  local results = {}
  local i = 1
  for dirname in lfs.dir('/mnt/saturation-records') do
    if dirname:gmatch(experiment_name)() then
      local constants, checkpoint = storage.load('/mnt/saturation-records/' .. dirname)
      local options = constants.options
      local test_losses = checkpoint.test_losses
      results[#results + 1] = {options=constants.options, train_losses=checkpoint.train_losses, test_losses=checkpoint.test_losses}
    end

    print(string.format('Processed directory %d', i))
    i = i + 1
  end

  return results
end

function extract_experiment_results(experiment_name)
  local results = load_experiment_results(experiment_name)
  torch.save('results/' .. experiment_name, results)
end

extract_experiment_results('saturation%-1')
