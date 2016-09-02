local SeqTakeLast, Module = torch.class('znn.SeqTakeLast', 'nn.Module')

function SeqTakeLast:__init(batchfirst)
    Module.__init(self)
    self.batchfirst = not not batchfirst

    self.gradInput = { self.gradInput, torch.Tensor() }
end

function SeqTakeLast:clearState()
    self.output:set()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    return self
end

function SeqTakeLast:updateOutput(input)
    local seq, length = input[1], input[2]

    if self.batchfirst then
        seq = seq:transpose(1,2)
    end
    local seqLen, bSize, dim = seq:size(1), seq:size(2), seq:size(3)

    assert( length:size(1) == bSize,
        "length not match:" .. bSize .. "/" .. length:size(1) )


    self.output = self.output or seq.new()
    self.output:typeAs(seq):resize(bSize, dim)

    self.output:view(1, bSize, dim):gather( 
      seq, 1, length:view(1, bSize, 1):expand(1,bSize,dim) )

    return self.output
end

function SeqTakeLast:updateGradInput(input, gradOutput)
    local seq, length = input[1], input[2]

    if self.batchfirst then
        seq = seq:transpose(1,2)
    end
    local seqLen, bSize, dim = seq:size(1), seq:size(2), seq:size(3)

    self.gradInput[1] = self.gradInput[1] or seq.new()

    assert( length:size(1) == bSize,  "length not match")
    assert( length:size(1) == gradOutput:size(1), "length not match")

    local gradInput = self.gradInput[1]
    gradInput:resizeAs(seq):zero()

    -- gI: seqLen x bSize x dim 
    -- gO: bSize  x dim

    gradInput:scatter(
      1, 
      length:view(1, bSize, 1):expand(1,bSize,dim),
      gradOutput:view(1, bSize, dim):expand( seqLen, bSize, dim )
    )

    self.gradInput[2]:resize(length:size()):zero()

    return self.gradInput
end
