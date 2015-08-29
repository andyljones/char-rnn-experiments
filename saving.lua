local torch = require 'torch'
local table = require 'std.table'
local lfs = require 'lfs'

local M = {}

function M.make_saver(options, alphabet, start_time)
  local start_date = os.date('%Y-%m-%d %H-%M-%S', start_time)
  local directory = string.format('checkpoints/%s', start_date)
  lfs.mkdir(directory)

  function saver(model, train_losses, test_losses)
    local checkpoint_num = table.size(test_losses)
    local best_so_far = test_losses[#train_losses]
    local filename = string.format('%s/%s-batches-%d-loss-%.2f.t7', directory, checkpoint_num, #train_losses, best_so_far)

    local checkpoint = {
      options=options,
      model=model,
      alphabet=alphabet,
      train_losses=train_losses,
      test_losses=test_losses
    }

    torch.save(filename, checkpoint)
  end

  return saver
end

return M
