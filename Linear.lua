local Linear, nnLinear = torch.class('znn.Linear', 'nn.Linear')

function Linear:__init(fromDim, toDim, hasbias)
    nnLinear.__init(self, fromDim, toDim, hasbias)
    self.fromDim = fromDim
    self.toDim = toDim
end

function Linear:updateOutput(input)
    local nDim = input:dim()
    if nDim <= 2 then
        nnLinear.updateOutput(self, input)
    else
        nnLinear.updateOutput(self, input:view(-1, self.fromDim))

        local pred_size = input:size()
        pred_size[nDim] = self.toDim
        self.output:resize(pred_size)
    end
    return self.output
end

function Linear:accGradParameters(input, gradOutput)
    nnLinear.accGradParameters(self,
        input:view(-1, self.fromDim),
        gradOutput:view(-1, self.toDim))
end

function Linear:updateGradInput(input, gradOutput)
    nnLinear.updateGradInput(self,
        input:view(-1, self.fromDim),
        gradOutput:view(-1, self.toDim))

    self.gradInput:resizeAs(input)
    return self.gradInput
end
