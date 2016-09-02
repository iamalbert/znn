#!/usr/bin/env th

local test = torch.TestSuite()

local tester = torch.Tester()


require 'cudnn'
require 'znn'

function test.SeqTakeLast()
    -- add test code here, using tester:eq methods

    local dim = 10
    local len = { 10, 5, 32, 17, 24 }

    local length = torch.LongTensor(len)
    local batchSize, maxLen = length:size(1), length:max()
    local seq = torch.rand( maxLen, batchSize, dim )

    local gradOutput = torch.rand( batchSize, dim )

    local targetGradInput = torch.zeros(maxLen, batchSize, dim)
    local target = torch.Tensor( batchSize, dim )
    for i = 1, batchSize do
      target[i]:copy( seq[{ len[i] , i }] )
      targetGradInput[{len[i], i}]:copy(gradOutput[i])
    end


    local input = { seq, length }

    local m = znn.SeqTakeLast()

    local pred = m:forward(input)

    tester:assertTensorEq( pred, target, "forward error")
    --- 

    local predGradInput = m:backward( input, gradOutput )
    tester:assertTensorEq( predGradInput[1], targetGradInput, 
      "backward error")
end



return tester:add(test):run()
