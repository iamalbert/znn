# znn
my extension to torch7, totally experimental and unstable

## Installation
git clone https://github.com/iamalbert/znn.git && cd znn && luarocks make

## Modules
* RNN Utitlies
  * PadToLongest: Take a table of tensor, output a padded version and the "length tensor".
  * TakeLastFrom: Take an `TxNxD` tensor and an "length tensor", output the corresponding posistion
  * SetPaddedValue: Take an `TxNxD` tensor and an "length tensor", set new value at the padded posistions. Useful in attention.
*  extension to `nngraph`
  * use `-` instead of nested parenthesis to construct your graph
*  extension to `nn`
  * patches to `clearState` problems in `nn.Index`, etc.

### PadToLongest

```lua
znn.PadToLongest( [value=0, [batchfirst=false]])
```
Accept a table of tensors {`T_1 x D`, `T_2 x D`, `T_3 x D`, ..., `T_N xD`}, output a table of 
 * a tensor `A` of the size `T x N x D`, where `T` is the maximum of `T_1`, `T_2`, ...
 * a tensor `L` recording the lengths of these tensor:
```lua
A[t][b][k] = { inputs[b][t][k]       if 1<= t <= inputs[b]:size(1)
             { value                 if     t >  inputs[b]:size(1)

L[b] = inputs[b]:size(1)
```
Example:
```lua
local m = znn.PadToLongest()
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs)
```
First, `output[1]` is a tensor of size `7x3x5` meaning a mini-batch of 3 sequences, that are padded to the length of 7, of vectors having dimension 5. 
Second, `output[2]` is `torch.LongTensor{ 4,7,5 }`.

Note that the size of second dimension of `inputs` must be consistent in a mini-batch, but can vary in another.

### TakeLastFrom
```lua
znn.TakeLastFrom([offset=0])
```
This module accepts a `TxNxD` tensor and a "length tensor" of size `T`, then output the vectors of last time step of each sequence in the minibatch. The size of the output is thus `NxD`.

Example:

```lua
local m = nn.Sequential()
  :add( znn.PadToLongest() )
  :add( nn.ParallelTable() 
    :add( cudnn.LSTM( 5, 10 ) )
    :add( nn.Identity() )
  )
  :add( znn.TakeLastFrom() )
  
local inputs = { torch.rand(4, 5),  torch.rand(7, 5), torch.rand(6, 5) }
local output = m:forward(inputs) -- a tensor of size 3x10, the last time step of each sequence in inputs.
```
Here we use the LSTM implemented in [cudnn.torch](https://github.com/soumith/cudnn.torch), you can replace it with `nn.SeqLSTM` from [rnn](https://github.com/Element-Research/rnn#rnn.SeqLSTM)
