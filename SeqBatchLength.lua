local SeqBatchLength, Module = torch.class('znn.SeqBatchLength', 'nn.Module')

function SeqBatchLength:__init(value) 
    Module.__init(self)

    self.output:long()
    self.gradInput = nil
end

function SeqBatchLength:clearState()
    self.output:set()
    return self
end

function SeqBatchLength:updateOutput(input)
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

    local output   = self.output

    if input[1]:type() == 'torch.CudaTensor' then
        output:cuda()
    else
        output:long()
    end
    length:resize(#input)

    for i=1, #input do
        output[i] = input[i]:size(1)
    end

    return self.output
end
