local Sub, nnModule = torch.class('znn.Sub', 'nn.Module')

local unpack = unpack or table.unpack

function Sub:__init(...)
	nnModule.__init(self)
    self.indices = {...}
end

function Sub:updateOutput(input)
    self.output = input:sub( unpack(self.indices) )
    return self.output
end

function Sub:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or input.new()

    self.gradInput:typeAs(input):resizeAs(input):zero()
    self.gradInput:sub( unpack(self.indices) ):copy(gradOutput)

    return self.gradInput
end

function Sub:__tostring()
	return torch.type(self) ..
		string.format( "(%s)", table.concat(self.indices, ", ") )
end

