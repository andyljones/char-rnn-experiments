local luaunit = require('luaunit')
local encoding = require('encoding')

function test_chars_to_ints()
  local text = 'aba'

  local actual_alphabet, actual_encoded = encoding.chars_to_ints(text)

  local expected_alphabet = {a=1, b=2}
  local expected_encoded = torch.Tensor{1, 2, 1}

  luaunit.assertTrue(torch.eq(expected_encoded, actual_encoded))
  luaunit.assertEquals(expected_alphabet, actual_alphabet)
end

function test_ints_to_one_hot()
  local ints = torch.Tensor{1, 2, 1}
  local width = 2

  local actual = encoding.ints_to_one_hot(ints, width)

  local expected = torch.Tensor{{1, 0}, {0, 1}, {1, 0}}

  luaunit.assertTrue(torch.eq(expected, actual))
end

function test_one_hot_to_ints()
  local one_hot = torch.Tensor{{0, 1}, {1, 0}}

  local actual = encoding.one_hot_to_ints(one_hot)

  local expected = torch.Tensor{2, 1}:long()

  luaunit.assertTrue(torch.eq(expected, actual))
end

function test_ints_to_chars()
  local alphabet = {a=1, b=2}
  local ints = torch.Tensor{1, 2, 1}

  local actual = encoding.ints_to_chars(alphabet, ints)

  local expected = 'aba'
  luaunit.assertEquals(actual, expected)
end

luaunit.LuaUnit.run()
