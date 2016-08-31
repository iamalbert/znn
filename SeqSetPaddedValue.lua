local SeqSetPaddedValue, nnModule = torch.class('znn.SeqSetPaddedValue', 'nn.Module')


function SeqSetPaddedValue:__init(value, inplace)
    assert( value ~= nil, "`value' shall be a number")

    self.value = value
    self.inplace = not not inplace 

    self.gradInput = { self.gradInput }
end

local fillPaddedValue = function(seq, length, dest, value)
    local max_len = seq:size(1)

    for i = 1, length:size(1) do
        local len = length[i]
        if len < max_len then
            dest:select(2, i):narrow(1, len+1, max_len - len):fill( value )
        end
    end
end

function SeqSetPaddedValue:updateOutput(input)
    local seq, length = input[1], input[2]

    self.output = self.output or seq.new() 
    local output = self.output

    output:typeAs(seq)
    if self.inplace then
        output:set(seq)
    else
        output:resizeAs(seq):copy(seq)
    end


    fillPaddedValue( seq, length, output, self.value )

    return self.output
end

function SeqSetPaddedValue:updateGradInput(input, gradOutput)
    local seq, length = input[1], input[2]

    self.gradInput[1] = self.gradInput[1] or gradOutput.new()
    local gradInput = self.gradInput[1]

    gradInput:typeAs(gradOutput)

    if self.inplace then
        gradInput:set(gradOutput)
    else
        gradInput:resizeAs(gradOutput):copy(gradOutput)
    end

    fillPaddedValue( seq, length, gradInput, 0 )

    return self.gradInput
end

function SeqSetPaddedValue:clearState()
  self.output:set()
  self.gradInput[1]:set()
  return self
end
