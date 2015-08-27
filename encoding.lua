local torch = require 'torch'
local table = require 'std.table'


function chars_to_ints(text)
  local alphabet = {}
  local encoded = torch.Tensor(#text)
  for i = 1, #text do
    local c = text:sub(i, i)
    if alphabet[c] == nil then alphabet[c] = table.size(alphabet) + 1 end
    local code = alphabet[c]
    encoded[i] = code
  end

  return alphabet, encoded
end

function ints_to_one_hot(ints, width)
  local height = ints:size()[1]
  local zeros = torch.zeros(height, width)
  local indices = ints:view(-1, 1):long()
  local one_hot = zeros:scatter(2, indices, 1)
  return one_hot
end

function one_hot_to_ints(one_hot)
  local _, ints = torch.max(one_hot, 1)
  return ints
end

function invert_alphabet(alphabet)
  local inverted = {}
  for char, code in pairs(alphabet) do
    inverted[code] = char
  end
  return inverted
end

function ints_to_chars(alphabet, ints)
  local decoder = invert_alphabet(alphabet)
  local decoded = {}
  for i = 1, ints:size(1) do
    decoded[i] = decoder[ints[i]]
  end

  return table.concat(decoded)
end

-- TESTS --
local luaunit = require('luaunit')

function test_chars_to_ints()
  local text = 'aba'

  local actual_alphabet, actual_encoded = chars_to_ints(text)

  local expected_alphabet = {a=1, b=2}
  local expected_encoded = torch.Tensor{1, 2, 1}

  luaunit.assertTrue(torch.eq(expected_encoded, actual_encoded))
  luaunit.assertEquals(expected_alphabet, actual_alphabet)
end

function test_ints_to_one_hot()
  local ints = torch.Tensor{1, 2, 1}
  local width = 2

  local actual = ints_to_one_hot(ints, width)

  local expected = torch.Tensor{{1, 0}, {0, 1}, {1, 0}}

  luaunit.assertTrue(torch.eq(expected, actual))
end

function test_one_hot_to_ints()
  local one_hot = torch.Tensor{{0, 1}, {1, 0}}

  local actual = one_hot_to_ints(one_hot)

  local expected = torch.Tensor{2, 1}:long()

  luaunit.assertTrue(torch.eq(expected, actual))
end

function test_ints_to_chars()
  local alphabet = {a=1, b=2}
  local ints = torch.Tensor{1, 2, 1}

  local actual = ints_to_chars(alphabet, ints)

  local expected = 'aba'
  luaunit.assertEquals(actual, expected)
end

luaunit.LuaUnit.run()
