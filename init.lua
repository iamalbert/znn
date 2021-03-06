znn = {
    __version = 1.0
}


do
    require 'nngraph'

    local Module = torch.getmetatable('nn.Module')
    Module.__unm__ = function( obj )
        return obj()
    end

    Module.__sub__ = function( prev, next )
        return next(prev)
    end


    local Node = torch.getmetatable('nngraph.Node')
    Node.__sub__ = function( prev, next )
        return next(prev)
    end
end

local files = {
    'util',
    'SeqBatchLength',
    'SeqPadding',
    'SeqTakeLast',
    'SeqSetPaddedValue',
    'SeqSetPaddedTargetDim',
    'CudnnRNN',
    'CudnnSeq2SeqDecoder',
    'CudnnGetStatesWrapper',
    'AdaptedLengthCriterion',
    'Linear',
    'Sub',
    'GetTableField',

    'CrossEntropyCriterion'

}

for _, file in ipairs(files) do
    torch.include('znn', file .. '.lua')
end
