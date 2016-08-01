local AdaptedLengthCriterion, Criterion = torch.class('znn.AdaptedLengthCriterion', 'nn.Criterion')


function AdaptedLengthCriterion:__init( criterion, lenp )
    self.criterion = criterion
    self.lenp = lenp
end


function AdaptedLengthCriterion:updateOutput( input, target )

    local len = math.min( input:size(1), target:size(1) )

    self.output = self.criterion:forward(
        input:narrow(1, 1, len) , 
        target:narrow(1, 1, len)
    )

    return self.output
end

function AdaptedLengthCriterion:updateGradInput(input, target)

    local len = math.min( input:size(1), target:size(1) )

    local gradInput = self.criterion:backward(
        input:narrow(1, 1, len) , 
        target:narrow(1, 1, len)
    )

    self.gradInput = self.gradInput or gradInput.new()
    self.gradInput:typeAs(gradInput):resizeAs(input):zero()
    self.gradInput:narrow(1, 1, len):copy(gradInput)

    return self.gradInput
end
