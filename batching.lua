local table = require 'std.table'
local math = require 'std.math'
local torch = require 'torch'
local encoding = require 'encoding'

local M = {}

function M.load_text()
  local f = io.open('input.txt')
  local text = f:read('*a')
  f:close()

  return text
end

function M.split_indices(indices, split_fractions)
  local split_sizes = (split_fractions*indices:size(1)):long()
  local split_points = torch.cat(torch.LongTensor{0}, split_sizes:cumsum())
  local splits = {}
  for i = 1, split_points:size(1) - 1 do
    local lower, upper = split_points[i] + 1, split_points[i+1]
    splits[i] = indices[{{lower, upper}}]
  end
  return splits
end

function M.make_chunk_iterator(encoded_text, indices, chunk_size, n_symbols)
  function co()
    for i = 1, math.huge do
      local index = indices[i%indices:size(1) + 1]
      local lower = (index - 1)*chunk_size + 1
      local upper = lower + chunk_size - 1
      local chunk = encoded_text[{{lower, upper}}]
      local X = encoding.ints_to_one_hot(encoded_text[{{lower, upper - 1}}], n_symbols)
      local y = encoded_text[{{lower + 1, upper}}]
      coroutine.yield(X, y)
    end
  end

  return coroutine.wrap(co)
end

function M.make_chunk_iterators(text, split_fractions, chunk_size)
  local alphabet, encoded_text = encoding.chars_to_ints(text)
  local n_chunks = math.floor(#text/chunk_size)

  local indices = torch.randperm(n_chunks)
  local splits = M.split_indices(indices, split_fractions)

  local iterators = {}
  for _, split in pairs(splits) do
    iterator = M.make_chunk_iterator(encoded_text, split, chunk_size, table.size(alphabet))
    iterators[#iterators + 1] = iterator
  end
  
  return alphabet, iterators
end

function stack(tensors)
  local shape = torch.totable(tensors[1]:size())
  table.insert(shape, 1, #tensors)

  local result = torch.Tensor(torch.LongStorage(shape))
  for i = 1, #tensors do
    result[i] = tensors[i]
  end

  return result
end

function M.make_batch_iterator(chunk_iterator, batch_size)
  function co()
    local Xs = {}
    local ys = {}
    while true do
      local X, y = chunk_iterator()
      if X and y then
        Xs[#Xs + 1] = X
        ys[#ys + 1] = y
      else
        break
      end

      if #Xs == batch_size then
        coroutine.yield(stack(Xs), stack(ys))
        Xs = {}
        ys = {}
      end
    end
  end

  return coroutine.wrap(co)
end

function M.make_batch_iterators(text, split_fractions, chunk_size, batch_size)
  local alphabet, chunk_iterators = M.make_chunk_iterators(text, split_fractions, chunk_size)

  local batch_iterators = {}
  for _, chunk_iterator in pairs(chunk_iterators) do
    batch_iterators[#batch_iterators + 1] =  M.make_batch_iterator(chunk_iterator, batch_size)
  end

  return alphabet, batch_iterators
end

return M
