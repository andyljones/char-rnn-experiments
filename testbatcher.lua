local luaunit = require('luaunit')
local batcher = require('batcher')

function test_split_indices()
  local indices = torch.Tensor{1, 2, 3, 4, 5}
  local fractions = torch.Tensor{0.4, 0.4, 0.2}

  local actual = batcher.split_indices(indices, fractions)

  local expected = {torch.Tensor{1, 2}, torch.Tensor{3, 4}, torch.Tensor{5}}

  luaunit.assertEquals(#actual, #expected)
  for i = 1, #expected do
    luaunit.assertTrue(torch.eq(actual[i], expected[i]))
  end
end

function test_make_chunk_iterators()
  local text = 'abbcbacc'
  local fractions = torch.Tensor{0.25, 0.75}

  local _, iterators = batcher.make_chunk_iterators(text, fractions, 2)

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

  local _, iterators = batcher.make_batch_iterators(text, fractions, 2, 2)

  local expected_lengths = {1, 3}
  for i = 1, #expected_lengths do
    local count = 0
    while iterators[i]() do count = count + 1 end
    luaunit.assertEquals(count, expected_lengths[i])
  end
end

luaunit.LuaUnit.run()
