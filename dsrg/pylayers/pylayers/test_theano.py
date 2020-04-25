import theano
import theano.tensor as T
from theano import pp
from theano import function
import numpy as np
from ipdb import set_trace


conv5 = T.ftensor4()
sim_map = T.ftensor3()
top_diff = T.ftensor4()

batch_size, c, h, w = conv5.shape
value = T.reshape(conv5, newshape=(batch_size, c, h*w))
value = T.transpose(value, axes=(0, 2, 1))

context = T.batched_dot(sim_map, value)
context = T.transpose(context, axes=(0, 2, 1))
context = T.reshape(context, newshape=(batch_size, c, h, w))

fuse = context + conv5

fuse_sum = T.sum(fuse * top_diff)

forward_theano = theano.function([conv5, sim_map], fuse)
backward_theano = theano.function([conv5, sim_map, top_diff], T.grad(fuse_sum, conv5))

one = np.ones(shape=(3, 3))
np_conv5 = np.stack([one, one + 1, one + 2, one + 3], axis=0).astype(np.float32)

sal = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
sal_reshape = np.tile(np.reshape(sal, (3 * 3, -1)), (1, 3 * 3))
sal_reshape_1 = np.transpose(sal_reshape)
np_sim = np.equal(sal_reshape, sal_reshape_1)
np_sim = np_sim.astype(np.float32)

temp = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
np_diff = np.stack([temp, temp, temp, temp], axis=0).astype(np.float32)

np_conv5 = np.expand_dims(np_conv5, axis=0)
np_sim = np.expand_dims(np_sim, axis=0)
np_diff = np.expand_dims(np_diff, axis=0)

cdiff = backward_theano(np_conv5, np_sim, np_diff)

# np_fuse = forward_theano(np_conv5, np_sim)

set_trace()

print("finished")
