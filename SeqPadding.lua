local SeqPadding, Module = torch.class('znn.SeqPadding', 'nn.Module')

function SeqPadding:__init(value, batchfirst)
    Module.__init(self)
    self.value = value or 0
    self.batchfirst = not not batchfirst
end

function SeqPadding:clearState()
    self.output:set()
    self.gradInput = {}
    return self
end

function SeqPadding:updateOutput(input)
    local longest = 0

    assert( #input > 0 , "input must be a table of tensors")

    for i = 1,#input do
        local len = input[i]:size(1)

        assert( input[i]:size(2) == input[1]:size(2),
            "inconsistent dimension within batch" )

        if len > longest then
            longest = len
        end
    end

    self.output = self.output or input[1].new()
    local output  = self.output

    -- seqLen x batch x dimension
    output:typeAs(input[1])

    if self.batchfirst then
        output:resize(#input, longest, input[1]:size(2) )
    else
        output:resize(longest, #input, input[1]:size(2) )
    end

    output:fill(self.value)

    for i=1, #input do
        local seq = input[i]
        if self.batchfirst then
            output:sub( 1, seq:size(1), i, i):copy(seq)
        else
            output:sub( i, i, 1, seq:size(1)):copy(seq)
        end
    end

    return self.output
end

function SeqPadding:updateGradInput(input, gradOutput)
    local gradInput = {}


    for i = 1, #input do
        if self.batchfirst then
            gradInput[i] = gradOutput[1]:sub(i, i, 1, input[i]:size(1))
        else
            gradInput[i] = gradOutput[1]:sub(1, input[i]:size(1), i, i)
        end
    end

    self.gradInput = gradInput
    return self.gradInput
end
