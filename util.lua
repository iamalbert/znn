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

    local ret

    local first = true

    for i, p in pairs(list) do
        if first then
            ret = znn.util.nestedMap( p, {}, function(old, new)
                return old.new():resize( 
                    len, table.unpack(old:size():totable()) )
            end)
            first = false
        end

        znn.util.nestedMap( p, ret, function(old, new)
            local dest = new[i]
            dest:copy(old)
        end)
    end

    return ret
end


