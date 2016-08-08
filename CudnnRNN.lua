
local RNN = cudnn.RNN

local origResetStates = RNN.resetStates
function RNN:resetStates()
   if self.hiddenInput then
      self.hiddenInput = nil
   end
   if self.cellInput then
      self.cellInput = nil
   end
   if self.gradHiddenOutput then
      self.gradHiddenOutput = nil
   end
   if self.gradCellOutput then
      self.gradCellOutput = nil
   end
end

local function getInputs(input)
    if torch.isTensor(input) then
        return input, nil, nil
    elseif type(input) == "table" then
        if #input == 2 then
            return input[2], input[1], nil
        elseif #input == 3 then
            return input[3], input[2], input[1]
        end
    end
    error "input shall be inputSeq or {initHidden,inputSeq}, or {initCell,initHidden,inputSeq}"
end

local function setOutputs(seq, hid, cell)
    assert( seq, "seq cannot be nil")
    if hid then
        if cell then
            return {cell, hid, seq}
        else
            return {hid, seq}
        end
    else
        return seq
    end
end

local CudnnRNN, Parent = torch.class('znn.CudnnRNN', 'nn.Container')

function CudnnRNN:__init(rnn)
    Parent.__init(self)
    assert( torch.isTypeOf(rnn, RNN), "expect an instance of cudnn.RNN")
    self.modules = {rnn}
    self.rnn = rnn
end

function CudnnRNN:updateOutput(input)
    local rnn = self.rnn
    local inputSeq, initHidden, initCell = getInputs(input)

    -- print{ inseq=inputSeq, initH=initHidden, initC=initCell }


    rnn.hiddenInput = initHidden
    rnn.cellInput   = initCell

    rnn:forward(inputSeq)


    self.output = setOutputs( rnn.output,
        initHidden and rnn.hiddenOutput:clone(), 
        initCell   and rnn.cellOutput:clone() 
    )

    return self.output
end

function CudnnRNN:updateGradInput(input, gradOutput)
    local inputSeq, initHidden, initCell = getInputs(input)
    local gradOutputSeq, gradHidden, gradCell = getInputs(gradOutput)

    local rnn = self.rnn


    rnn.hiddenInput = initHidden
    rnn.cellInput   = initCell

    rnn.gradHiddenOutput = gradHidden
    rnn.gradCellOutput   = gradCell

    local gradInputSeq = rnn:updateGradInput(inputSeq, gradOutputSeq)

    self.gradInput = setOutputs( 
        gradInputSeq, 
        initHidden and rnn.gradHiddenInput:clone(), 
        initCell   and rnn.gradCellInput:clone() 
    )

    return self.gradInput
end


function CudnnRNN:accGradParameters(input, gradOutput, scale)
    local inputSeq, initHidden, initCell = getInputs(input)
    local gradOutputSeq, gradHidden, gradCell = getInputs(gradOutput)


    local rnn = self.rnn

    rnn.hiddenInput = initHidden
    rnn.initCell    = initCell

    rnn.gradHiddenOutput = gradHidden
    rnn.gradCellOutput   = gradCell

    rnn:accGradParameters( inputSeq, gradOutputSeq, scale)

    return self
end


local LSTM, Parent = torch.class('znn.CudnnLSTM', 'znn.CudnnRNN')
function LSTM:__init(...)
    Parent.__init( self, cudnn.LSTM(...) )
end

