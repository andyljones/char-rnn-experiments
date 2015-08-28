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
  local _, ints = torch.max(one_hot, 2)
  return ints:view(-1)
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

function one_hot_to_chars(alphabet, one_hot)
  return ints_to_chars(alphabet, one_hot_to_ints(one_hot))
end

return {
  chars_to_ints=chars_to_ints,
  ints_to_one_hot=ints_to_one_hot,
  ints_to_chars=ints_to_chars,
  one_hot_to_ints=one_hot_to_ints,
  one_hot_to_chars=one_hot_to_chars
}
