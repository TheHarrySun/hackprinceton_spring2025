import tensorflow as tf
from . import inits


_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
    
def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + "_" + str(get_layer_uid(layer))
            
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.isspare = False
    
    def _call(self, inputs):
        return inputs
    
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs
        
        
class GraphConvolutionSparseMulti(MultiLayer):
    def __init__(self, input_dim, output_dim, adj_mats, nonzero_feat, dropout=0, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(input_dim[self.edge_type[1]], output_dim, name='weights_%d' % k)
                
    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat[self.edge_type[1]])
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
            
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l1_normalize(outputs, dim=1)
        return outputs
    
    
class GraphConvolutionMulti(MultiLayer):
    def __init__(self, input_dim, output_dim, adj_mats, dropout = 0, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(input_dim, output_dim, name='weights_%d' % k)
        
    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs