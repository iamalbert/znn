local CrossEntropyCriterion, Criterion = torch.class('znn.CrossEntropyCriterion', 'nn.Criterion')

function CrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.ClassNLLCriterion(weights)
end

function CrossEntropyCriterion:updateOutput(input, target)

   if input:dim() > 1 then
     input = input:view(-1, input:size( input:dim() ) )
   end

   target = type(target) == 'number' and target or target:view(-1)

   self.lsm:updateOutput(input)
   self.nll:updateOutput(self.lsm.output, target)
   self.output = self.nll.output
   return self.output
end

function CrossEntropyCriterion:updateGradInput(input, target)
   local size = input:size()

   if input:dim() > 1 then
     input = input:view(-1, input:size( input:dim() ) )
   end
   target = type(target) == 'number' and target or target:view(-1)

   self.nll:updateGradInput(self.lsm.output, target)
   self.lsm:updateGradInput(input, self.nll.gradInput)
   self.gradInput:view(self.lsm.gradInput, size)
   return self.gradInput
end

return nn.CrossEntropyCriterion
