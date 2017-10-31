from tensorflow.python.ops.rnn_cell import BasicLSTMCell, LSTMStateTuple, RNNCell, GRUCell
from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.ops.variable_scope import variable_scope
from tensorflow.python.framework.ops import name_scope
from tensorflow.contrib.framework import nest
from tensorflow import ones_initializer

class GatedFeedbackLSTMCell(BasicLSTMCell):
    def __init__(self, num_units, layer_pos, forget_bias=1.0, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(GatedFeedbackLSTMCell, self).__init__(num_units=num_units, forget_bias=forget_bias, state_is_tuple=True, activation=activation, reuse=reuse)
        self._layer_pos = layer_pos
        self._kernel_initializer = kernel_initializer
        self._bias_initializer= bias_initializer
    def call(self, inputs, state):
        if not nest.is_sequence(state):
            raise ValueError("Expected state to be a tuple of length %d, but receive: %s" % (len(self.state_size), state))
        n_layer = len(state)
        c, h = state[self._layer_pos]
        concat_h = array_ops.concat([s[-1] for s in state], axis=1)

        with variable_scope('input-forget-output-gates'):
            conc = _linear([inputs, h], 3 * self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
            i, f, o = array_ops.split(conc, 3, axis=1)
        with variable_scope('scalar-gates'):
            gates = _linear([inputs, concat_h], n_layer, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        with variable_scope('gated-inputs'):
            gated_h = \
                _linear(array_ops.reshape(array_ops.expand_dims(gates, axis=2) * array_ops.expand_dims(h, axis=1), (-1, n_layer * self._num_units)),
                        self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        with variable_scope('new-inputs'):
            new_inputs = \
                self._activation(_linear(inputs, self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer) + gated_h)
        new_c = self._activation(c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * new_inputs)
        new_h = new_c * math_ops.sigmoid(o)
        new_state = LSTMStateTuple(new_c, new_h)
        return (new_h, new_state)

class GatedFeedbackGRUCell(GRUCell):
    def __init__(self, num_units, layer_pos, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(GatedFeedbackGRUCell, self).__init__(num_units=num_units, activation=activation, reuse=reuse, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        self._layer_pos = layer_pos
    def call(self, inputs, state):
        n_layer = len(state)
        h = state[self._layer_pos]
        concat_h = array_ops.concat(state, axis=1)

        with variable_scope('reset-output-gates'):
            dtype = next(a.dtype for a in [inputs, h])
            bias_ones = ones_initializer(dtype=dtype) if self._bias_initializer is None else self._bias_initializer
            value = math_ops.sigmoid(_linear([inputs, h], 2 * self._num_units, True, bias_ones, self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with variable_scope('scalar-gates'):
            gates = _linear([inputs, concat_h], n_layer, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer)
        with variable_scope('gated-inputs'):
            gated_h = \
                _linear(array_ops.reshape(array_ops.expand_dims(gates, axis=2) * array_ops.expand_dims(h, axis=1), (-1, n_layer * self._num_units)),
                        self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer) * r
        with variable_scope('new-inputs'):
            new_inputs = \
                self._activation(_linear(inputs, self._num_units, True, bias_initializer=self._bias_initializer, kernel_initializer=self._kernel_initializer) + gated_h)
        new_h = u * h + (1 - u) * new_inputs
        return new_h, new_h

class MultiGatedFeedbackRNNCell(RNNCell):
    def __init__(self, cells):
        super(MultiGatedFeedbackRNNCell, self).__init__()
        if not cells:
            raise ValueError("Must specify at least one cell for MultiGatedFeedbackRNNCell")
        if not nest.is_sequence(cells):
            raise TypeError("cells must be a list of tuple, but saw: %s." % cells)
        self._cells = cells
    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)
    @property
    def output_size(self):
        return self._cells[-1].output_size
    def zero_state(self, batch_size, dtype):
        with name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
    def call(self, inputs, state):
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with variable_scope('cell_%d' % i):
                if not nest.is_sequence(state):
                    raise ValueError('Expected state to be a tuple of length %d, but received: %s' % (len(self.state_size), state))
                cur_inp, new_state = cell(cur_inp, state)
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states