znn = {
    __version = 1.0
}

local files = {
    'SeqBatchLength',
    'SeqPadding'
}

for _, file in ipairs(files) do
    torch.include('znn', file .. '.lua')
end
