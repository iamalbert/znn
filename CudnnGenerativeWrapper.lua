local CudnnGenerativeWrapper, nnContainer = torch.class('znn.CudnnGenerativeWrapper', 'nn.Container')

function CudnnGenerativeWrapper:__init( rnn, generateLen )
    assert( cudnn ~= nil, "require cudnn.torch")
    assert( torch.isTypeOf(rnn, cudnn.RNN), "only support subclass of cudnn.RNN ")

    generateLen = tonumber(generateLen) or 0
    assert( generateLen > 0, "generateLen should be a number greater than 0")

    self.modules = { rnn }
    self.generateLen = generateLen
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
    local initVector, initHidden, initCell = input[1], input[2], input[3]
    rnn.hiddenInput = initHidden
    rnn.cellInput   = initCell

    local generateLen = self.generateLen
    local rnn = self.modules[1]

    self.buffer = self.buffer or initVector.new()
    local buffer = self.buffer

    local bufferSize = torch.LongStorage{
        1+self.generateLen, initVector:size(2), initVector:size(3)
    }
    buffer:typeAs(initVector):resize(bufferSize)

    buffer[1]:copy(initVector)

    self.bufferX = buffer:narrow(1, 1, generateLen)
    self.output:set( buffer:narrow(1, 2, generateLen) )

    rnn.output:set(self.output)
    rnn:forward(self.bufferX)

    return self.output
end

function CudnnGenerativeWrapper:updateGradInput(input, gradOutput)
    local initVector, initHidden, initCell = input[1], input[2], input[3]
    rnn.hiddenInput = initHidden
    rnn.cellInput   = initCell

    self.gradInput = rnn:backward( self.bufferX, gradOutput )
    return self.gradInput
end
