local CudnnGetStatesWrapper, nnContainer = torch.class('znn.CudnnGetStatesWrapper', 'nn.Container')

function CudnnGetStatesWrapper:__init(rnn)
    nnContainer.__init(self)
    self.modules = {rnn}

    self.output = {}
end

function CudnnGetStatesWrapper:updateOutput(input)
    local rnn = self.modules[1]
    local pred = rnn:forward(input)

    self.output = {
        -- pred:narrow(1, input:size(1), 1):clone(),
        rnn.hiddenOutput:clone(),
        rnn.cellOutput:clone()
    }
    return self.output
end

function CudnnGetStatesWrapper:updateGradInput(input, gradOutput)
    local rnn = self.modules[1]

    self.buffer = self.buffer or torch.CudaTensor()
    local buffer = self.buffer

    buffer:resizeAs(rnn.output):zero()

    -- buffer[ input:size(1) ]:copy(gradOutput[1])

    rnn.gradHiddenOutput = gradOutput[1]
    rnn.gradCellOutput   = gradOutput[2]

    self.gradInput = rnn:backward(input, buffer)

    return self.gradInput
end
