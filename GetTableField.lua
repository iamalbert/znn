local GetTableField, nnModule = torch.class('znn.GetTableField', 'nn.Module')

function GetTableField:__init( k1, k2, k3, k4 )
  assert( k1 ~= nil, "key cannot be nil")

  nnModule.__init(self)
  self.k1 = k1
  self.k2 = k2
  self.k3 = k3
  self.k4 = k4

  self.output = nil
  self.gradInput = nil
end

function GetTableField:updateOutput( input )
  self.output = input[ self.k1 ]

  if self.k2 then self.output = self.output[ self.k2 ] end
  if self.k3 then self.output = self.output[ self.k3 ] end
  if self.k4 then self.output = self.output[ self.k4 ] end

  return self.output
end

function GetTableField:clearStates()
  self.output = nil
  self.gradInput = nil
end
