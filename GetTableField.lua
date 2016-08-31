local GetTableField, nnModule = torch.class('znn.GetTableField', 'nn.Module')

function GetTableField:__init( key )
  assert( key ~= nil, "key cannot be nil")

  nnModule.__init(self)
  self.key = key

  self.output = nil
  self.gradInput = nil
end

function GetTableField:updateOutput( input )
  self.output = input[ self.key ]
  return self.output
end

function GetTableField:clearStates()
  self.output = nil
  self.gradInput = nil
end
