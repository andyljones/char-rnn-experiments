local table = require 'std.table'
local training = require 'training'
local lfs = require 'lfs'
local storage = require 'storage'
local torch = require 'torch'
local cjson = require 'cjson'
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
            split = torch.Tensor({0.95, 0.05}),
            name = 'saturation-2',
            save_dir = 'saturation-records'
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
  local param_gen = make_param_generator({3, 50, 21}, {1, 100, 21})
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
  for dirname in lfs.dir('saturation-records') do
    if dirname:gmatch(experiment_name)() then
      local constants, checkpoint = storage.load('saturation-records/' .. dirname)
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

function load_results(experiment_name)
  return torch.load('results/' .. experiment_name)
end

function results_as_json(experiment_name)
  local results = load_experiment_results(experiment_name)
  local losses = {}
  for _, r in pairs(results) do
    local n_timesteps, n_neurons = r.options.n_timesteps, r.options.n_neurons
    local loss = r.test_losses[#r.test_losses]
    losses[#losses+1] = {n_timesteps=n_timesteps, n_neurons=n_neurons, loss=loss}
  end

  local f = io.open(string.format('results/%s.json', experiment_name), 'w')
  f:write(cjson.encode(losses))
  f:close()
end

run()
-- results_as_json('saturation%-1')
