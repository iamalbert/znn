# znn
my extension to torch7, totally experimental and unstable

## Installation
Using luarocks
```
git clone https://github.com/iamalbert/znn.git && cd znn && luarocks make
```

## Modules
* RNN Utitlies
  * SeqPadding: Take a table of sequence (2D tensor), output a padded version.
  * SeqBatchLength: Take a table of sequence (2D tensor), output the lengths of all sequences. 
  * SeqTakeLast: Take an `TxNxD` tensor and an "length tensor", output the corresponding posistion
  * SeqSetPaddedValue: Take an `TxNxD` tensor and an "length tensor", set new value at the padded posistions. Useful in attention.
*  extension to `nngraph`
  * use `-` instead of nested parenthesis to construct your graph
*  extension to `nn`
  * patches to `clearState` problems in `nn.Index`, etc.

### SeqPadding
```lua
znn.SeqPadding( [value = 0 [,batchfirst = false]])
```
Accept a table of tensors {`T_1 x D`, `T_2 x D`, `T_3 x D`, ..., `T_N xD`}, output a tensor `A` of the size `T x N x D`, where `T` is the maximum of `T_1`, `T_2`, ...
```lua
A[t][n][k] = { inputs[n][t][k]       if 1<= t <= inputs[n]:size(1)
             { value                 if     t >  inputs[n]:size(1)
```
for ` 1 <= t <= T, 1<= n <= N, 1 <= k <= D`

Example:
```lua
local m = znn.SeqPadding()
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs)
```
`output` is a tensor of size `7x3x5` meaning a mini-batch of 3 sequences, that are padded to the length of 7, of vectors having dimension 5. 

Note that the size of second dimension of `inputs` must be consistent in a mini-batch, but can vary in another.


###  SeqBatchLength
```lua
znn.SeqBatchLength()
```
Accept a table of tensors {`T_1 x D`, `T_2 x D`, `T_3 x D`, ..., `T_N xD`}, output a tensor `L` recording the lengths of these tensor: 
```lua
L[n] = inputs[n]:size(1)
```
for ` 1<= n <= N`


Example:
```lua
local m = znn.SeqBatchLength()
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs)
```
Second, `output` is `torch.LongTensor{ 4,7,6 }`.

Note that the size of second dimension of `inputs` must be consistent in a mini-batch, but can vary in another.


### SeqTakeLast
```lua
znn.SeqTakeLast([offset = 0 [,batchfirst = false]])
```
This module accepts a `TxNxD` tensor and a "length tensor" of size `T`, then output the vectors of last time step of each sequence in the minibatch. The size of the output is thus `NxD`.

Example:
```lua
local S = torch.randn(7,3,5),
local L = torch.LongTensor{4,7,6}
local output = znn.SeqTakeLast():forward{ S, L }

print(output)

-- output[1] == S[ L[1] ][ 1 ]
-- output[2] == S[ L[2] ][ 2 ]
-- output[3] == S[ L[3] ][ 3 ]
```


```lua
local m = nn.Sequential()
  :add( nn.ConcatTable()
    :add( nn.Sequential()
       :add( znn.SeqPadding() )
       :add( cudnn.LSTM( 5, 10 ) )
    )
    :add( nn.SeqBatchLength() )
  )
  :add( znn.SeqTakeLast() )
  
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs) -- a tensor of size 3x10, the last time step of each sequence in inputs.
```

Here we use the LSTM implemented in [cudnn.torch](https://github.com/soumith/cudnn.torch), you can replace it with `nn.SeqLSTM` from [rnn](https://github.com/Element-Research/rnn#rnn.SeqLSTM)

The above example is more clear if we writing using `nngraph`:
```lua
local create_model = function(inDim, outDim)

  local input   = - nn.Identity()
  local seq     = input - znn.SeqPadding() - cudnn.LSTM( inDim, outDim )
  local seqLen  = input - znn.SeqBatchLength()
  local seqRepr = { seq, seqLen } -  znn.SeqTakeLast()
  
  return nn.gModule( {input}, {seqRepr} )
end

local m = create_model(5, 10)
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs) -- a tensor of size 3x10, the last time step of each sequence in inputs.
```
