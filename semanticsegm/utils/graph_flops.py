

import tensorflow as tf
from collections import defaultdict

def graph_flops(g = None, batch = 1, format = 'NCHW', targets = None,
                training = True, verbose = False, sess_config = None):
    if g is None:
        g = tf.get_default_graph()
    if targets:
        reachable = set()
        todo = targets
        while todo:
            t = todo.pop()
            o = t.op
            if o in reachable:
                continue
            reachable.add(o)
            #print "{} -> {}".format(t.name, o.name)
            for i in o.inputs:
                #print "{} --> {}".format(o.name, i.name)
                if 'gradients' in i.name.split('/'):
                    continue
                todo.append(i)
    else:
        reachable = None
    types = defaultdict(int)
    flops = 0
    for o in g.get_operations():
        if reachable and (o not in reachable):
            #print "skipping {}".format(o.name)
            continue
        #if o.name.startswith('gradients'):
        #    continue
        #print dir(o)
        #print o.type
        types[o.type] += 1
        if o.type == 'Conv2D':
            #print o.name, o.inputs[0], o.inputs[1], o.outputs[0]
            #print o
            fmt = o.get_attr('data_format').decode("utf-8")
            if format != fmt:
                print('WARNING: tensor format ({}) does not match expected ({})'.format(fmt, format))
            strides = o.get_attr('strides')
            padding = o.get_attr('padding').lower()
            in_size = o.inputs[0].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            if fmt == 'NCHW':
              in_size = tuple(in_size)
              strides = tuple((int(strides[2]), int(strides[3])))
            else:
              in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
              strides = tuple((int(strides[1]), int(strides[2])))
            w_size = o.inputs[1].shape.as_list()
            filter_sz = tuple((w_size[0], w_size[1]))
            channels_in = w_size[2]
            channels_out = w_size[3]
            assert(channels_in == in_size[1])
            flops += (2.0 *
                      in_size[0] * # batch
                      (in_size[2] / strides[0]) * # image width
                      (in_size[3] / strides[1]) * # image height
                      in_size[1] * channels_out * # channels in/out
                      filter_sz[0] * filter_sz[1])
            if verbose:
                print("_CONV Input={} ChannelsOut={} Filters={} Stride={} Padding='{}' # {}".format(in_size,
                                                                                                    channels_out,
                                                                                                    filter_sz,
                                                                                                    strides,
                                                                                                    padding,
                                                                                                    o.name))
        if o.type == 'Conv2DBackpropInput':
            #print o.name, o.inputs[2], o.inputs[1], o.outputs[0]
            fmt = o.get_attr('data_format').decode("utf-8")
            #assert(fmt == format)
            if format != fmt:
                print('WARNING: tensor format ({}) does not match expected ({})'.format(fmt, format))
            strides = o.get_attr('strides')
            padding = o.get_attr('padding').lower()
            in_size = o.inputs[2].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            if fmt == 'NCHW':
              in_size = tuple(in_size)
              strides = tuple((int(strides[2]), int(strides[3])))
            else:
              in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
              strides = tuple((int(strides[1]), int(strides[2])))
            w_size = o.inputs[1].shape.as_list()
            filter_sz = tuple((w_size[0], w_size[1]))
            channels_in = w_size[3]
            channels_out = w_size[2]
            assert(channels_in == in_size[1])
            flops += (2.0 *
                      in_size[0] * # batch
                      in_size[2] * # image width (stride cancels out)
                      in_size[3] * # image height
                      in_size[1] * channels_out * # channels in/out
                      filter_sz[0] * filter_sz[1])
            if verbose:
                print("_DECONV Input={} ChannelsOut={} Filters={} Stride={} Padding='{}' # {}".format(in_size,
                                                                                                      channels_out,
                                                                                                      filter_sz,
                                                                                                      strides,
                                                                                                      padding,
                                                                                                      o.name))
        if o.type == 'Relu':
            in_size = o.inputs[0].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            else:
                if in_size[0] != batch:
                    print("WARNING: batch size appears to be {}, not {}".format(in_size[0],
                                                                                batch))
            if format == 'NCHW':
                in_size = tuple(in_size)
            else:
                in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
            if verbose:
                print("_RELU Input={} # {}".format(in_size,
                                                   o.name))
        if o.type in ('FusedBatchNorm', 'FusedBatchNormV2'):
            in_size = o.inputs[0].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            else:
                if in_size[0] != batch:
                    print("WARNING: batch size appears to be {}, not {}".format(in_size[0],
                                                                                batch))
            if format == 'NCHW':
                in_size = tuple(in_size)
            else:
                in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
            if verbose:
                print("_BATCH_NORM Input={} axis=1 # {}".format(in_size,
                                                                o.name))
        if o.type == 'RealDiv':
            if o.name.endswith('/dropout/div'):
                in_size = o.inputs[0].shape.as_list()
                if in_size[0] is None:
                    in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
                else:
                    if in_size[0] != batch:
                        print("WARNING: batch size appears to be {}, not {}".format(in_size[0],
                                                                                batch))
                if format == 'NCHW':
                    in_size = tuple(in_size)
                else:
                    in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
                if verbose:
                    print("_DROPOUT Input={} # {}".format(in_size,
                                                          o.name))
                continue
        if o.type == 'Softmax':
            #print o.name, o.inputs[0]
            #print o.inputs[0].op
            #print o.inputs[0].op.inputs[0]
            in_size = o.inputs[0].op.inputs[0].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            else:
                if in_size[0] != batch:
                    print("WARNING: batch size appears to be {}, not {}".format(in_size[0],
                                                                                batch))
            if format == 'NCHW':
                in_size = tuple(in_size)
            else:
                in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
            if verbose:
                print("_SOFTMAX Input={} # {}".format(in_size,
                                                      o.name))
        if o.type == 'ResizeBilinear':
            #print o.name, o.inputs[0], o.inputs[1], o.outputs[0]
            in_size = o.inputs[0].shape.as_list()
            if in_size[0] is None:
                in_size = [ batch, in_size[1], in_size[2], in_size[3] ]
            else:
                if in_size[0] != batch:
                    print("WARNING: batch size appears to be {}, not {}".format(in_size[0],
                                                                                batch))
            if format == 'NCHW':
                in_size = tuple(in_size)
            else:
                in_size = tuple((in_size[0], in_size[3], in_size[1], in_size[2]))
            # kind of annoying - we have to evaluate the second input to
            #  determine the output tensor size
            out_size = tf.Session(config=sess_config).run(o.inputs[1])

            # each output pixel is assumed to require 11 flops (2 multiplies
            #  for each of the 2x2 input pixel window + 3 additions)
            flops += (in_size[0] *
                      in_size[1] *
                      out_size[0] *
                      out_size[1] *
                      11)

    if training:
        flops *= 3

    if verbose:
        print("# total {} FLOPS = {:f}".format(('training' if training else 'inference'),
                                               flops))
    #for k in sorted(types):
    #    print k, types[k]

    return flops
