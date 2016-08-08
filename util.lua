znn.util.template_until_tensor = function(obj, new, f)
    if torch.isTensor(obj) then
        new = new or obj.new()
        f(obj, new)
        return new
    elseif torch.type(obj) == "table" then -- is an table
        new = new or {}
        for k,v in pairs(obj) do
            new[k] = zd.util.template_until_tensor(v, new[k], f)
        end
        return new
    else
        error "expecting tensors or nested table of tensors"
    end
end
