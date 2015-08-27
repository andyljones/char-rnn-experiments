local table = require 'std.table'
local math = require 'std.math'
local torch = require 'torch'

function load_text()
  local f = io.open('input.txt')
  local text = f:read('*a')
  f:close()

  return text
end

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

function generate_chunks(encoded_text, chunk_size)
  n_chunks = math.floor(encoded_text:size()[1]/chunk_size)
  indices =  torch.randperm(n_chunks)

  function co()
    for i = 1, n_chunks do
      local index = indices[i]
      local lower = (index - 1)*chunk_size + 1
      local upper = lower + chunk_size - 1
      chunk = encoded_text[{{lower, upper}}]
      coroutine.yield(chunk)
    end
  end

  return coroutine.create(co)
end

-- local text = load_text()
-- local alphabet, encoded = chars_to_ints(text)
-- local chunk_generator = generate_chunks(encoded, 10)
-- print(coroutine.resume(chunk_generator))
-- print(coroutine.resume(chunk_generator))

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

luaunit.LuaUnit.run()
