require 'cudnn'
require 'znn'

local m = znn.CudnnGetStatesWrapper( cudnn.LSTM(5, 5, 1) )
local input = torch.rand(7, 2, 5):cuda()
local gradOutput = {
    torch.rand(1, 2, 5):cuda(),
    torch.rand(1, 2, 5):cuda(),
}


local pred = m:forward(input)
local gradInput = m:backward(input, gradOutput)
--[[
print{
    pred = pred,
    grad = gradInput
}
print( gradInput )

local len = 4
local bs = 2
local nLayer = 2
local rnn  = znn.CudnnLSTM(5, 5, nLayer)
local m = znn.CudnnSeq2SeqDecoder( rnn, len+2 )

m:training()

local p1 = m:forward( torch.rand(len, bs, 5):cuda() )
print(p1)

print("--")
local p1 = m:forward {
    torch.rand(nLayer, bs, 5):cuda(),
    torch.rand(len, bs, 5):cuda(),
}
print(p1[1])
print(p1[2])

print("--")
m:evaluate()
local p1 = m:forward( torch.rand(1, bs, 5):cuda() )
print(p1)

local p1 = m:forward {
    torch.rand(nLayer, bs, 5):cuda(),
    torch.rand(1, bs, 5):cuda(),
}
print(p1)
--]]

local p =  torch.rand(3,4,5,6):split(1) 
print(p)
print( znn.util.nestedJoin(p) )
