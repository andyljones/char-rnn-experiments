local table = require 'std.table'
local math = require 'std.math'
local torch = require 'torch'

function load_text()
  local f = io.open('input.txt')
  local text = f:read('*a')
  f:close()

  return text
end

function make_chunk_iterator(encoded_text, indices, chunk_size, n_symbols)
  function co()
    for i = 1, indices:size(1) do
      local index = indices[i]
      local lower = (index - 1)*chunk_size + 1
      local upper = lower + chunk_size - 1
      local chunk = ints_to_one_hot(encoded_text[{{lower, upper}}], n_symbols)
      coroutine.yield(chunk)
    end
  end

  return coroutine.wrap(co)
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

function make_chunk_iterators(text, split_fractions, chunk_size)
  local alphabet, encoded_text = chars_to_ints(text)
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


-- TESTS --
function test_split_indices()
  local indices = torch.Tensor{1, 2, 3, 4, 5}
  local fractions = torch.Tensor{0.4, 0.4, 0.2}

  local actual = split_indices(indices, fractions)

  local expected = {torch.Tensor{1, 2}, torch.Tensor{3, 4}, torch.Tensor{5}}

  luaunit.assertEquals(#actual, #expected)
  for i = 1, #expected do
    luaunit.assertTrue(torch.eq(actual[i], expected[i]))
  end
end

function test_make_chunk_iterators()
  local text = 'abbcbacc'
  local fractions = torch.Tensor{0.25, 0.75}

  local _, iterators = make_chunk_iterators(text, fractions, 2)

  local expected_lengths = {1, 3}
  for i = 1, #expected_lengths do
    local count = 0
    while iterators[i]() do count = count + 1 end
    luaunit.assertEquals(count, expected_lengths[i])
  end
end

function test_make_batch_iterators()
  local text = 'abbcbaccabbcbacc'
  local fractions = torch.Tensor{0.25, 0.75}

  local _, iterators = make_batch_iterators(text, fractions, 2, 2)

  local expected_lengths = {1, 3}
  for i = 1, #expected_lengths do
    local count = 0
    while iterators[i]() do count = count + 1 end
    luaunit.assertEquals(count, expected_lengths[i])
  end
end

-- 
-- luaunit.LuaUnit.run()
