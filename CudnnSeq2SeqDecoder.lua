local CudnnSeq2SeqDecoder, Parent = torch.class('znn.CudnnSeq2SeqDecoder', 'nn.Container')

function CudnnSeq2SeqDecoder:__init( net, generateLen, out2in )

    Parent.__init(self)

    self.net = net
    self.modules = { net }
    self.generateLen = generateLen

    if out2in then
        assert( type(out2in) == "function", "out2in shall be a function" )
        self.out2in = out2in
    end

end

function CudnnSeq2SeqDecoder:setLength(length)
    self.generateLen = length
end

function CudnnSeq2SeqDecoder:out2in(output)
    return output
end

function CudnnSeq2SeqDecoder:updateOutput(input)
    local net = self.net


    if self.train then

        local o = net:updateOutput(input)
        self.output = o

    else
        local outs = {}
        local generateLen = self.generateLen

        for i = 1, generateLen do
            outs[i] = nn.utils.recursiveCopy({}, net:forward( input ) )

            input = self:out2in(outs[i])
        end

        self.output = outs
    end

    return self.output
end

function CudnnSeq2SeqDecoder:backward(input, gradOutput)
    self.gradInput = self.net:backward( input, gradOutput )
    return self.gradInput
end

function CudnnSeq2SeqDecoder:updateGradInput(input, gradOutput)
    self.gradInput = self.net:updateGradInput(input, gradOutput)
    return self.gradInput
end

function CudnnSeq2SeqDecoder:accGradParameters(input, gradOutput, scale)
    self.net:accGradParameters(input, gradOutput, scale)
    return self
end
