local SeqSetPaddedTargetDim, nnModule = torch.class('znn.SeqSetPaddedTargetDim', 'nn.Module')


function SeqSetPaddedTargetDim:__init(targetDim, inplace)
    assert( targetDim ~= nil, "`targetDim' shall be a number")

    self.targetDim = targetDim
    self.inplace = not not inplace 

    self.gradInput = { self.gradInput }
end

local fillPaddedTargetDim = function(seq, length, dest, targetDim, value)
    local max_len = seq:size(1)

    for i = 1, length:size(1) do
        local len = length[i]
        if len < max_len then
            local tmp = dest
                :select(2, i)                     -- seqLen x dim
                :narrow(1, len+1, max_len - len)  -- paddedLen x dim
                :fill( 0 )                        -- 

            if targetDim and value then
                tmp
                :narrow(2, targetDim, 1)             -- paddedLen x 1 (targetDimDim)
                :fill(value)
            end
        end
    end
end

function SeqSetPaddedTargetDim:updateOutput(input)
    local seq, length = input[1], input[2]

    self.output = self.output or seq.new() 
    local output = self.output

    output:typeAs(seq)
    if self.inplace then
        output:set(seq)
    else
        output:resizeAs(seq):copy(seq)
    end


    fillPaddedTargetDim( seq, length, output, self.targetDim, 1e8 )

    return self.output
end

function SeqSetPaddedTargetDim:updateGradInput(input, gradOutput)
    local seq, length = input[1], input[2]

    self.gradInput[1] = self.gradInput[1] or gradOutput.new()
    local gradInput = self.gradInput[1]

    gradInput:typeAs(gradOutput)

    if self.inplace then
        gradInput:set(gradOutput)
    else
        gradInput:resizeAs(gradOutput):copy(gradOutput)
    end

    fillPaddedTargetDim( seq, length, gradInput )

    return self.gradInput
end

function SeqSetPaddedTargetDim:clearState()
  self.output:set()
  self.gradInput[1]:set()
  return self
end
