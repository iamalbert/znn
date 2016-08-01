local SeqBatchLength, Module = torch.class('znn.SeqBatchLength', 'nn.Module')

function SeqBatchLength:__init(dim) 
    Module.__init(self)

    self.dim = dim or 1

    self.output:long()
    self.gradInput = nil
end

function SeqBatchLength:clearState()
    self.output:set()
    return self
end

function SeqBatchLength:updateOutput(input)


    assert( #input > 0 , "input must be a table of tensors")

    local dim = self.dim
    local output = self.output

    if input[1]:type() == 'torch.CudaTensor' then
        output:cuda()
    else
        output:long()
    end

    output:resize(  #input )

    for i=1, #input do
        assert( input[i]:size(dim) == input[i]:size(dim),
            "inconsistent dimension within batch" )
        output[i] = input[i]:size(dim)
    end

    return self.output
end
