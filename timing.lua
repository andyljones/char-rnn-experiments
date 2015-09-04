
local M = {}

function M.make_timer()
  local start_time = os.clock()
  local calls = 0
  function co()
    calls = calls + 1
    local time_total = os.clock() - start_time
    local time_per = time_total/calls
    return time_per
  end

  return co
end

return M
