znn = {
    __version = 1.0
}

local files = {

}

for _, file in ipairs(files) do
    torch.include('znn', file .. '.lua')
end
