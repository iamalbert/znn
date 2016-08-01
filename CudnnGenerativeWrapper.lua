local CudnnGenerativeWrapper, Parent = torch.class('znn.CudnnGenerativeWrapper', 'nn.Module')

function CudnnGenerativeWrapper:__init( rnn, generateLen )

    Parent.__init(self)

    assert( cudnn ~= nil, "require cudnn.torch")
    assert( torch.isTypeOf(rnn, cudnn.RNN), "only support subclass of cudnn.RNN ")

    generateLen = tonumber(generateLen) or 0
    assert( generateLen > 0, "generateLen should be a number greater than 0")

    self.modules = { rnn }
    self.generateLen = generateLen
    
    self.bias = torch.CudaTensor(rnn.hiddenSize)
    self.gradBias = torch.CudaTensor(rnn.hiddenSize)
    self:reset()

    self.gradInput = {}
end

function CudnnGenerativeWrapper:reset(stdv)
    stdv = stdv or 1.0 / math.sqrt(self.modules[1].hiddenSize)
    self.bias:uniform(-stdv, stdv)
    self.gradBias:zero()
end

function joinTensorTable( tensors, dest )
    local t = tensors[1]
    dest:typeAs(t):resizeAs( #tensors, t:size(2), t:size(3) )
    for i, tensor in ipairs(tensors) do
        dest[i]:copy( tensor )
    end
    return dest
end

function CudnnGenerativeWrapper:updateOutput(input)
    local rnn = self.modules[1]
    local initHidden, initCell = input[1], input[2]


    local generateLen = self.generateLen

    self.buffer = self.buffer or torch.CudaTensor()
    self.buffer:resize( generateLen+1, initHidden:size(2), initHidden:size(3) )

    self.output = self.buffer:narrow(1, 2, generateLen)
    local output = self.output


    local inputCurr = torch.CudaTensor( initHidden:size() )

    for i = 1, initHidden:size(2) do
        inputCurr[{1,i}]:copy( self.bias )
    end

    self.hidden = { [0] = initHidden:clone() }
    self.cell   = { [0] = initCell and initCell:clone()  }

    rnn.hiddenInput = self.hidden[0]
    rnn.cellInput   = self.cell[0]

    rnn.rememberStates = true
    for t = 1, generateLen do

        local pred = rnn:forward( inputCurr )

        output:select(1,t):copy(pred)

        self.hidden[t] = rnn.hiddenOutput:clone()
        self.cell[t]   = rnn.cellOutput:clone()

        inputCurr = output:narrow(1, t, 1)
    end

    rnn.rememberStates = false
    --[[
    rnn.hiddenInput = self.hidden[0]
    rnn.cellInput   = self.cell[0]
    rnn:forward( self.buffer:narrow(1, 1, generateLen) )
    --]]

    return self.output
end

function CudnnGenerativeWrapper:backward(input, gradOutput)
    local rnn = self.modules[1]
    local initHidden, initCell = input[1], input[2]

    local generateLen = self.generateLen
    local gradFirst

    for t = generateLen, 1, -1 do

        rnn.output:copy( self.output:narrow(1,t,1) )
        rnn.hiddenInput = self.hidden[t-1]
        rnn.cellInput   = self.cell[t-1]

        gradFirst = rnn:backward( self.buffer:narrow(1, t, 1), gradOutput:narrow(1,t,1) )

        rnn.gradHiddenOutput = rnn.gradHiddenInput:clone()
        rnn.gradCellOutput   = rnn.gradCellInput:clone()
    end
    --]]

    --[[
    gradFirst = rnn:backward( self.buffer:narrow(1, 1, generateLen), gradOutput )
    --]]
    
    self.gradBias:add( gradFirst[1]:sum(1):view(-1) )

    self.gradInput[1] = rnn.gradHiddenInput:clone()
    self.gradInput[2] = rnn.gradCellInput:clone()

    return self.gradInput
end

