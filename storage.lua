local torch = require 'torch'
local table = require 'std.table'
local lfs = require 'lfs'
local string = require 'std.string'
require 'nngraph'

local M = {}

function M.make_saver(model, options, alphabet, start_time)
  local start_date = os.date('%Y-%m-%d %H-%M-%S', start_time)
  local directory = string.format('%s/%s-%s', options.save_dir, options.name, start_date)
  lfs.mkdir(directory)

  local filename = string.format('%s/constants.t7', directory)
  torch.save(filename, {model=model, alphabet=alphabet, options=options})

  function saver(params, train_losses, test_losses)
    local checkpoint_num = table.size(test_losses)
    local best_so_far = test_losses[#test_losses]
    local filename = string.format('%s/%s-batches-%d-loss-%.2f.t7', directory, checkpoint_num, #train_losses, best_so_far)

    local checkpoint = {
      params=params,
      train_losses=train_losses,
      test_losses=test_losses
    }

    torch.save(filename, checkpoint)
  end

  return saver
end

function M.load(datestring)
  local directory = string.format('checkpoints/%s', datestring)
  local constants = torch.load(directory .. '/constants.t7')

  local checkpoints = {}
  for filename in lfs.dir(directory) do
    local number = filename:gmatch('(%d+)%-')()
    if number then
      checkpoints[tonumber(number)] = filename
    end
  end

  local latest_checkpoint = torch.load(string.format('%s/%s', directory, checkpoints[#checkpoints]))

  return constants, latest_checkpoint
end

function M.load_model(datestring)
  local constants, checkpoint = M.load(datestring)
  constants.model.params:copy(checkpoint.params)

  return constants.model, constants.alphabet, constants.options
end

function M.load_options_and_params(datestring)
  local constants, checkpoint = M.load(datestring)

  return constants.options, params
end

return M
