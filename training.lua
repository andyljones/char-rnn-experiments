local batcher = require 'batcher'
local gru = require 'gru'
local encoding = require 'encoding'
local torch = require 'torch'
local table = require 'std.table'

function decode(alphabet, batch)
  local results = {}
  for i = 1, batch:size(1) do
    results[i] = encoding.one_hot_to_chars(alphabet, batch[i])
  end

  return results
end

function batch_one_hot_to_ints(batch)
  return select(2, batch:max(3)):view(batch:size(1), batch:size(2))
end

function calculate_loss(output, input)
    local n_samples, n_timesteps, n_symbols = unpack(torch.totable(input:size()))

    local relevant_outputs = output[{{}, {1, -2}}]:clone():view(-1, n_symbols)
    local targets = batch_one_hot_to_ints(input)[{{}, {2, -1}}]:clone():view(-1)

    local criterion = nn.ClassNLLCriterion()
    local loss = criterion:forward(relevant_outputs, targets)
    local grad_loss = criterion:backward(relevant_outputs, targets):view(n_samples, n_timesteps - 1, n_symbols)
    local grad_loss = torch.cat(grad_loss, torch.zeros(n_samples, 1, n_symbols), 2)

    return loss, grad_loss
end

function make_feval(model, n_neurons)
  function feval(input)
    local n_samples = input:size(1)
    local initial_state = torch.zeros(n_samples, n_neurons)
    local final_state = torch.zeros(n_samples, n_neurons)

    local output, final_state = unpack(model:forward({input, initial_state}))
    local loss, grad_loss = calculate_loss(output, input)
    local grad_input = model:backward({input, initial_state}, {grad_loss, final_state})

    return model:getParameters()
  end
end


local n_neurons, n_timesteps, n_samples = 20, 10, 2

local text = batcher.load_text()
local alphabet, batch_iterator = batcher.make_batch_iterators(text, torch.Tensor{1}, n_timesteps, n_samples)

local initial_state = torch.zeros(n_samples, n_neurons)
local final_state = torch.zeros(n_samples, n_neurons)
local model = gru.build(n_timesteps, table.size(alphabet), n_neurons)

local input = batch_iterator[1]()

local output, final_state = unpack(model:forward({input, initial_state}))
local loss, grad_loss = calculate_loss(output, input)
local grad_input = model:backward({input, initial_state}, {grad_loss, final_state})
-- flat_params, flat_grad_params = model:getParameters()
-- print(flat_grad_params:size())
local params, grad_params = model:getParameters()
print(params:size()[1])
print(table.size(alphabet))
