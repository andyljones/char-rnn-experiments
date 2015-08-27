local luaunit = require('luaunit')
local encoding = require('encoding')

function test_chars_to_ints()
  local text = 'aba'

  local actual_alphabet, actual_encoded = encoding.chars_to_ints(text)

  local expected_alphabet = {a=1, b=2}
  local expected_encoded = torch.Tensor{1, 2, 1}

  luaunit.assertTrue(torch.eq(actual_encoded, expected_encoded))
  luaunit.assertEquals(actual_alphabet, expected_alphabet)
end

function test_ints_to_one_hot()
  local ints = torch.Tensor{1, 2, 1}
  local width = 2

  local actual = encoding.ints_to_one_hot(ints, width)

  local expected = torch.Tensor{{1, 0}, {0, 1}, {1, 0}}

  luaunit.assertTrue(torch.eq(actual, expected))
end

function test_one_hot_to_chars()
  local alphabet = {a=1, b=2}
  local one_hot = torch.Tensor{{0, 1}, {1, 0}}

  local actual = encoding.one_hot_to_chars(alphabet, one_hot)

  local expected = 'ba'

  luaunit.assertEquals(actual, expected)
end

luaunit.LuaUnit.run()
