local SeqTakeLast, Module = torch.class('SeqTakeLast', 'nn.Module')

function SeqTakeLast:__init(batchfirst)
    Module.__init(self)
    self.batchfirst = not not batchfirst

    self.gradInput = { self.gradInput }
end

function SeqTakeLast:clearState()
    self.output:set()
    self.gradInput[1]:set()
    return self
end

function SeqTakeLast:updateOutput(input)
    local seq, length = input[1], input[2]
    -- assume seqLen x batch x dimension

    if self.batchfirst then
        seq = seq:transpose(1,2)
    end

    self.output = self.output or seq.new()
    self.output:typeAs(seq):resize(seq:size(2), seq:size(3))

    assert( length:size(1) == seq:size(2), 
        "length not match:" .. seq:size(2) .. "/" .. length:size(1) )

    for i = 1, length:size(1) do
        self.output[i]:copy( seq[{length[i], i}] )
    end

    return self.output
end

function SeqTakeLast:updateGradInput(input, gradOutput)
    local seq, length = input[1], input[2]

    if self.batchfirst then
        seq = seq:transpose(1,2)
    end

    self.gradInput[1] = self.gradInput[1] or seq.new()

    local gradInput = self.gradInput[1]
    gradInput:typeAs(seq):resizeAs(seq):zero()

    assert( length:size(1) == seq:size(2), "length not match")
    assert( length:size(1) == gradOutput:size(1), "length not match")

    for i = 1, length:size(1) do
        gradInput[{length[i], i}]:copy(gradOutput[i])
    end

    return self.gradInput
end
