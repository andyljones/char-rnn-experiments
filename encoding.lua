local torch = require 'torch'
local table = require 'std.table'

local M = {}

function M.chars_to_ints(text)
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

function M.ints_to_one_hot(ints, n_symbols)
  local height = ints:size()[1]
  local zeros = torch.zeros(height, n_symbols)
  local indices = ints:view(-1, 1):long()
  local one_hot = zeros:scatter(2, indices, 1)
  return one_hot
end

function M.chars_to_one_hot(alphabet, text)
  local ints = torch.Tensor(#text)
  for i = 1, #text do
    local c = text:sub(i, i)
    ints[i] = alphabet[c]
  end

  return M.ints_to_one_hot(ints, table.size(alphabet))
end

function M.one_hot_to_ints(one_hot)
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

function M.ints_to_chars(alphabet, ints)
  local decoder = invert_alphabet(alphabet)
  local decoded = {}
  for i = 1, ints:size(1) do
    decoded[i] = decoder[ints[i]]
  end

  return table.concat(decoded)
end

function M.one_hot_to_chars(alphabet, one_hot)
  return M.ints_to_chars(alphabet, M.one_hot_to_ints(one_hot))
end

return M
