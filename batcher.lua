local table = require 'std.table'
local math = require 'std.math'
local torch = require 'torch'
local encoding = require 'encoding'

function load_text()
  local f = io.open('input.txt')
  local text = f:read('*a')
  f:close()

  return text
end

function split_indices(indices, split_fractions)
  local split_sizes = (split_fractions*indices:size(1)):long()
  local split_points = torch.cat(torch.LongTensor{0}, split_sizes:cumsum())
  local splits = {}
  for i = 1, split_points:size(1) - 1 do
    local lower, upper = split_points[i] + 1, split_points[i+1]
    splits[i] = indices[{{lower, upper}}]
  end
  return splits
end

function make_chunk_iterator(encoded_text, indices, chunk_size, n_symbols)
  function co()
    for i = 1, indices:size(1) do
      local index = indices[i]
      local lower = (index - 1)*chunk_size + 1
      local upper = lower + chunk_size - 1
      local chunk = encoding.ints_to_one_hot(encoded_text[{{lower, upper}}], n_symbols)
      coroutine.yield(chunk)
    end
  end

  return coroutine.wrap(co)
end

function make_chunk_iterators(text, split_fractions, chunk_size)
  local alphabet, encoded_text = encoding.chars_to_ints(text)
  local n_chunks = math.floor(#text/chunk_size)

  local indices = torch.randperm(n_chunks)
  local splits = split_indices(indices, split_fractions)

  local iterators = {}
  for _, split in pairs(splits) do
    iterator = make_chunk_iterator(encoded_text, split, chunk_size, table.size(alphabet))
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

function make_batch_iterator(chunk_iterator, batch_size)
  function co()
    local batch = {}
    while true do
      local chunk = chunk_iterator()
      if chunk then
        batch[#batch + 1] = chunk
      else
        break
      end

      if #batch == batch_size then
        coroutine.yield(stack(batch))
        batch = {}
      end
    end
  end

  return coroutine.wrap(co)
end

function make_batch_iterators(text, split_fractions, chunk_size, batch_size)
  local alphabet, chunk_iterators = make_chunk_iterators(text, split_fractions, chunk_size)

  local batch_iterators = {}
  for _, chunk_iterator in pairs(chunk_iterators) do
    batch_iterators[#batch_iterators + 1] =  make_batch_iterator(chunk_iterator, batch_size)
  end

  return alphabet, batch_iterators
end


return {
  load_text=load_text,
  make_batch_iterators=make_batch_iterators,
  testing = {
    make_chunk_iterators=make_chunk_iterators,
    split_indices=split_indices
  }
}
