#!/usr/bin/env th

local test = torch.TestSuite()

local tester = torch.Tester()

require 'cudnn'
require 'znn'

function test.SeqTakeLast()
    -- add test code here, using tester:eq methods

    local makeTest = function( name, LongTensor, DoubleTensor )
      local dim = 50
      local len = { 10, 5, 32, 17, 24, 12, 2, 39, 25 }

      local length = torch.LongTensor(len):type(LongTensor)
      local batchSize, maxLen = length:size(1), length:max()

      local seq = torch.rand(maxLen,batchSize,dim):type(DoubleTensor)
      local gradOutput = torch.rand(batchSize,dim):type(DoubleTensor)

      local targetGradInput = seq.new( seq:size() ):zero()
      local target          = seq.new( gradOutput:size() )

      for i = 1, batchSize do
        target[i]:copy( seq[{ len[i] , i }] )
        targetGradInput[{len[i], i}]:copy(gradOutput[i])
      end


      local input = { seq, length }

      local m = znn.SeqTakeLast():type(DoubleTensor)

      local pred = m:forward(input)

      tester:eq( pred, target, name .. " forward error")
      --- 

      local predGradInput = m:backward( input, gradOutput )
      tester:eq( predGradInput[1], targetGradInput, name .. " backward error")
    end

    makeTest("cpu", "torch.LongTensor", "torch.DoubleTensor")
    makeTest("gpu", "torch.CudaTensor", "torch.CudaTensor")
end



return tester:add(test):run()
