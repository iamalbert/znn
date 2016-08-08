local util = {}
znn.util = util

util.nestedMap = function(obj, new, f)
    if torch.isTensor(obj) or torch.isStorage(obj) then
        new = new or obj.new()
        local p = f(obj, new)
        if p then new = p end
        return new
    elseif torch.type(obj) == "table" then -- is an table
        new = new or {}
        for k,v in pairs(obj) do
            local p = util.nestedMap(v, new[k], f)
            if p then new[k] = p end
        end
        return new
    else
        error "expecting tensors or nested table of tensors"
    end
end

util.nestedJoin = function( list )

    local len = #list

    local ret = znn.util.nestedMap( list[1], {}, function(old, new)
        return old.new():resize( 
            len, table.unpack(old:size():totable()) )
    end)

    for i, p in pairs(list) do
        znn.util.nestedMap( p, ret, function(old, new)
            local dest = new[i]

            --[[
            local ns = table.concat(dest:size():totable(), 'x')
            local os = table.concat(old:size():totable(), 'x')
            --]]

            dest:copy(old)
        end)
    end

    return ret
end


