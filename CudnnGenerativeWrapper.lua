local CudnnGenerativeWrapper, Parent = torch.class('znn.CudnnGenerativeWrapper', 'nn.Module')

function CudnnGenerativeWrapper:__init( rnn, generateLen, initTokenIndex )

    Parent.__init(self)

    assert( cudnn ~= nil, "require cudnn.torch")
    assert( torch.isTypeOf(rnn, cudnn.RNN), "only support subclass of cudnn.RNN ")

    generateLen = tonumber(generateLen) or 0
    assert( generateLen > 0, "generateLen should be a number greater than 0")

    assert( initTokenIndex and initTokenIndex > 0, "init token should be number > 0")
    self.initTokenIndex = initTokenIndex

    assert( rnn.bidirectional == 'CUDNN_UNIDIRECTIONAL', "only support uni-directional rnn")
    assert( not rnn.batchFirst, "only support rnn in seqLen x batchSize x inputSize")

    self.modules = { rnn }
    self.generateLen = generateLen
    
    self.gradInput = {}
    self.output = torch.CudaTensor()
    self:training()
end

function CudnnGenerativeWrapper:o2i(output, next_input)
    local maxval, maxidx = output:max(2)
    next_input:scatter(2, maxidx:cuda(), 1)
end

local function joinTensorTable( tensors, dest )
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
    local numLayers = rnn.numLayers 

    local batchSize = initHidden:size(2)
    local hiddenSize = rnn.hiddenSize
    local inputSize = rnn.inputSize

    assert( hiddenSize == initHidden:size(3), "incorrect input hidden size")
    assert( numLayers  == initHidden:size(1), "initHidden should be numLayers x batch x hiddenSize")

    self.inputBuffer = self.inputBuffer or torch.CudaTensor()
    self.inputBuffer:resize( generateLen, batchSize, inputSize):zero()
    local inputBuffer = self.inputBuffer

    self.output = self.output or torch.CudaTensor()
    self.output:resize( generateLen, batchSize, hiddenSize )
    local output = self.output

    inputBuffer[{1,{}, self.initTokenIndex}] = 1 

    self.hidden = { [0] = initHidden:clone() }
    self.cell   = { [0] = initCell and initCell:clone()  }
    self.reserved = {}

    rnn.rememberStates = false
    for t = 1, generateLen do

        local currInput = inputBuffer:narrow(1, t, 1)
        local currOutput = output[t]

        rnn.hiddenInput = self.hidden[t-1]
        rnn.cellInput   = self.cell[t-1]

        local pred = rnn:forward( currInput )

        currOutput:copy(pred)

        self.hidden[t]   = rnn.hiddenOutput:clone()
        self.cell[t]     = rnn.cellOutput:clone()
        self.reserved[t] = rnn.reserve:clone()

        if t ~= generateLen then
            self:o2i( currOutput, inputBuffer[t+1] )
        end
    end


    --[[
    if self.train then
        rnn.hiddenInput = self.hidden[0]
        rnn.cellInput   = self.cell[0]
        rnn:forward( self.InputBuffer:narrow(1, 1, generateLen) )
    end
    --]]

    return self.output
end

function CudnnGenerativeWrapper:backward(input, gradOutput)
    local rnn = self.modules[1]
    local initHidden, initCell = input[1], input[2]

    local generateLen = self.generateLen

    local batchSize = initHidden:size(2)
    local hiddenSize = rnn.hiddenSize
    local inputSize = rnn.inputSize
    local numLayers = rnn.numLayers 

    assert( hiddenSize == initHidden:size(3), "incorrect input hidden size")
    assert( numLayers  == initHidden:size(1), "initHidden should be numLayers x batch x hiddenSize")

    for t = generateLen, 1, -1 do

        rnn.output:copy( self.output:narrow(1,t,1) )
        rnn.hiddenInput = self.hidden[t-1]
        rnn.cellInput   = self.cell[t-1]
        rnn.reserve:copy(self.reserved[t])

        rnn:backward(
            self.inputBuffer:narrow(1, t,1),
            self.output:narrow(1, t, 1)
        )

        rnn.gradHiddenOutput = rnn.gradHiddenInput:clone()
        rnn.gradCellOutput   = rnn.gradCellInput:clone()
    end
    --]]

    --[[
    rnn.hiddenInput = self.hidden[0]
    rnn.cellInput   = self.cell[0]
    gradFirst = rnn:backward( self.InputBuffer:narrow(1, 1, generateLen), gradOutput )
    --]]

    self.gradInput[1] = rnn.gradHiddenOutput
    self.gradInput[2] = initCell and rnn.gradCellOutput

    return self.gradInput
end

