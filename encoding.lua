local torch = require 'torch'
local table = require 'std.table'

local M = {}

function M.chars_to_ints(text)
  local alphabet = {}
  local probs = {}
  local encoded = torch.Tensor(#text)
  for i = 1, #text do
    local c = text:sub(i, i)
    if alphabet[c] == nil then
      alphabet[#alphabet+1] = c
      alphabet[c] = #alphabet
      probs[c] = 0
    end
    encoded[i] = alphabet[c]
    probs[c] = probs[c] + 1/#text
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

function M.batch_ints_to_one_hot(ints, n_symbols)
  local batch_size, seq_length = ints:size(1), ints:size(2)
  local zeros = torch.zeros(batch_size, seq_length, n_symbols)
  local indices = torch.Tensor(batch_size, seq_length, 1):copy(ints):long()
  local one_hot = zeros:scatter(3, indices, 1)
  return one_hot
end


function M.chars_to_one_hot(alphabet, text)
  local ints = torch.Tensor(#text)
  for i = 1, #text do
    local c = text:sub(i, i)
    ints[i] = alphabet[c]
  end

  return M.ints_to_one_hot(ints, #alphabet)
end

function M.one_hot_to_ints(one_hot)
  local _, ints = torch.max(one_hot, 2)
  return ints:view(-1)
end

function M.ints_to_chars(alphabet, ints)
  local decoded = {}
  for i = 1, ints:size(1) do
    decoded[i] = alphabet[ints[i]]
  end

  return table.concat(decoded)
end

function M.one_hot_to_chars(alphabet, one_hot)
  return M.ints_to_chars(alphabet, M.one_hot_to_ints(one_hot))
end

return M
