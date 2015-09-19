local encoding = require 'encoding'
local torch = require 'torch'

local M = {}

function M.make_batch_iterators(chunk_size, batch_size)
  assert(chunk_size%2 == 1, 'Chunk size is not odd; can\'t create mirror batches')
  local alphabet = {a=1, b=2, ['|']=3, [1]='a', [2]='b', [3]='|'}
  local seq_length = (chunk_size - 1)/2

  function reverse(batch)
    local reversed = torch.Tensor(batch:size())
    for i = 1, batch:size(2) do reversed[{{}, i}] = batch[{{}, batch:size(2) - i + 1}] end
    return reversed
  end

  function co()
    for i = 1, math.huge do
      local seqs = torch.Tensor(batch_size, seq_length):random(1, 2)
      local seps = torch.Tensor(batch_size, 1):fill(3)
      local ints = torch.cat(torch.cat(seqs, seps, 2), reverse(seqs), 2)

      local X = ints[{{}, {1, chunk_size - 1}}]
      local y = ints[{{}, {2, chunk_size}}]

      coroutine.yield(encoding.batch_ints_to_one_hot(X, 3), y)
    end
  end

  local batch_iterators = {coroutine.wrap(co), coroutine.wrap(co)}

  return alphabet, batch_iterators
end

function M.error_bound(chunk_size)
  local seq_length = (chunk_size - 1)/2
  local n_unpredictable = seq_length - 1 -- we don't try and predict the first symbol
  local n_predicted = chunk_size - 1

  return math.log(2)*n_unpredictable/n_predicted
end

return M
