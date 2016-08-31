local util = {}
znn.util = util

local function isTensorOrStorage(obj)
  return torch.isTensor(obj) or torch.isStorage(obj)
end

util.nestedMap = function(obj, new, f, key)
    if isTensorOrStorage(obj) then
        new = new or obj.new()
        local p = f(obj, new, key)
        if p ~= nil then new = p end
        return new
    elseif torch.type(obj) == "table" then -- is an table
        new = new or {}
        for k,v in pairs(obj) do
            local p = util.nestedMap(v, new[k], f, k)
            if p ~= nil then new[k] = p end
        end
        return new
    else
        error "expecting tensors or nested table of tensors"
    end
end

util.nestedMap2 = function(obj1, obj2, new, f, key)
    local t1, t2 = torch.type(obj1), torch.type(obj2)
    -- print(t1, obj1, t2, obj2)

    -- if t1 ~= t2 then error(("encounter two different types: %s and %s"):format(t1, t2)) end

    if isTensorOrStorage(obj1) then
        new = new or obj1.new()
        local p = f(obj1, obj2, new, key)
        if p ~= nil then new = p end
        return new
    elseif torch.type(obj1) == "table" then -- is an table
      local t2 = torch.type(obj2)
      assert( t2 == "table", "obj1 is a table, but obj2 is " .. t2 )

      new = new or {}
      for k,v in pairs(obj1) do
        local p = util.nestedMap2(obj1[k], obj2[k], new[k], f, k)
          if p ~= nil then new[k] = p end
        end
      return new

    else
        error "expecting tensors or nested table of tensors"
    end
end

util.nestedJoin = function( list, joinDim )

    local len = #list

    local ret
    local first = true


    for i, p in pairs(list) do
        if first then
          joinDim = joinDim or util.nestedMap( p, nil, 
            function(o, n) return 1 end)

          ret = znn.util.nestedMap2( p, joinDim, {}, 
            function(old, dim, new)
              local size = old:size():totable()
              table.insert(size, dim, len )
              return old.new():resize( table.unpack(size) )
            end
          )
          first = false
        end

        znn.util.nestedMap2( p, joinDim, ret, 
          function(old, dim, new)
            new:narrow(dim, i, 1):copy(old)
          end
        )
    end

    return ret
end


