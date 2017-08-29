'''
Models for composition into a full structure.
'''


import theano
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams()


# ADAM stochastic gradient descent

def adam(loss, params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=(1 - 1e-8)):
    grads = T.grad(loss, params)
    updates = []
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1 * gamma ** (t - 1)
    for theta_previous, g in zip(params, grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        m = b1_t * m_previous + (1 - b1_t) * g
        v = b2 * v_previous + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1))
    return updates


# KL divergence from a factored normal distribution to the standard normal distribution.

def kld_to_std_normal(mean, ls):
    return 0.5 * (mean.shape[1] * ((2 * ls).exp() - 1 - 2 * ls) + T.square(mean).sum(1))


# Log density per dimension of a sample under a factored normal distribution.

def normal_log_density(sample, mean, ls):
    return (-0.5 * T.square(sample - mean) * (-2 * ls).exp() - ls - 0.5 * np.log(2 * np.pi)).sum(-1)


# Rectified linear unit

def relu(x):
    return T.maximum(x, 0)


# Convolutional Neural Network

class CNN():

    def __init__(self, x_depth, y_depth, kernel_size=3, upscale=1, downscale=1, hidden_layers=2, h_depth=256, res_depth=0, output='linear', name=''):
        if not output in ['linear', 'relu']: raise Exception('Invalid output type.')
        layer_depths = [x_depth] + [h_depth] * hidden_layers + [y_depth]
        self.upscale, self.downscale = upscale, downscale
        self.layer_kernels = [theano.shared(self._init_kernel(layer_depths[i], layer_depths[i + 1], kernel_size), name=('%s_kernel%d' % (name, i))) for i in range(len(layer_depths) - 1)]
        self.res_kernels = [[[theano.shared(self._init_res_kernel(layer_depths[i], kernel_size), name=('%s_kernel%d_res%d' % (name, i, ii))) for iii in range(2)] for ii in range(res_depth)] for i in range(len(layer_depths))]
        self.layer_biases = [theano.shared(np.zeros(layer_depths[i + 1], theano.config.floatX), name=('%s_bias%d' % (name, i))) for i in range(len(layer_depths) - 1)]
        self.output = output

    def __call__(self, x):
        for k1, k2 in self.res_kernels[0]: x += T.nnet.conv2d(relu(T.nnet.conv2d(x, k1, border_mode='half')), k2, border_mode='half')
        for i, (k, b) in enumerate(zip(self.layer_kernels, self.layer_biases)):
            if i > 0: x = relu(x)
            if self.upscale > 1: x = x.repeat(self.upscale, axis=2)[:, :, :-1].repeat(self.upscale, axis=3)[:, :, :, :-1]
            x = T.nnet.conv2d(x, k, subsample=(self.downscale, self.downscale), border_mode='half') + T.shape_padright(b, 2)
            for k1, k2 in self.res_kernels[i + 1]: x += T.nnet.conv2d(relu(T.nnet.conv2d(x, k1, border_mode='half')), k2, border_mode='half')
        if self.output == 'linear': return x
        elif self.output == 'relu': return relu(x)

    def _init_kernel(self, in_depth, out_depth, kernel_size, scale=1.):
        return np.random.normal(scale=(scale * np.sqrt(2. / (in_depth * kernel_size ** 2 + out_depth))), size=(out_depth, in_depth, kernel_size, kernel_size)).astype(theano.config.floatX)

    def _init_res_kernel(self, depth, kernel_size):
        return self._init_kernel(depth, depth, kernel_size, scale=0.01)

    def get_params(self):
        return self.layer_kernels + [aaa for a in self.res_kernels for aa in a for aaa in aa] + self.layer_biases


# Multilayer Perceptron.

class MLP():

    def __init__(self, n_x, n_y, hidden_layers=1, n_h=256, res_depth=0, output='linear', softmax_scale=1., dropout=None, dropout_output=False, weight_norm=False, name=''):
        if not output in ['linear', 'relu', 'softmax']: raise Exception('Invalid output type')
        if dropout != None and not (dropout >= 0. and dropout < 1): raise Exception('Dropout must be in the range [0, 1)')
        layer_sizes = [n_x] + [n_h] * hidden_layers + [n_y]
        self.layer_weights = [theano.shared(self._init_weight(*layer_sizes[i:(i + 2)]), name=('%s_weights%d' % (name, i))) for i in range(len(layer_sizes) - 1)]
        self.layer_weight_norms = [T.sqrt(T.square(a).sum()) for a in self.layer_weights] if weight_norm else [None] * len(self.layer_weights)
        self.layer_weight_gs = [theano.shared(np.ones((), dtype=theano.config.floatX), name=('%s_weights%d' % (name, i))) for i in range(len(layer_sizes) - 1)] if weight_norm else [None] * len(self.layer_weights)
        self.res_weights = [[[theano.shared(self._init_res_weight(layer_sizes[i]), name=('%s_weights%d_res%d' % (name, i, ii))) for iii in range(2)] for ii in range(res_depth)] for i in range(len(layer_sizes))]
        self.layer_biases = [theano.shared(np.zeros(layer_sizes[i + 1], theano.config.floatX), name=('%s_bias%d' % (name, i))) for i in range(len(layer_sizes) - 1)]
        self.output = output
        self.softmax_scale = softmax_scale
        self.dropout = dropout
        self.dropout_output = dropout_output
        self.weight_norm = weight_norm

    def __call__(self, x, rng=None, mode=None):
        for w1, w2 in self.res_weights[0]: x += relu(x.dot(w1)).dot(w2)
        for i, (w, w_norm, w_g, b) in enumerate(zip(self.layer_weights, self.layer_weight_norms, self.layer_weight_gs, self.layer_biases)):
            if i > 0: x = relu(x)
            if self.weight_norm: x = x.dot(w_g * w / w_norm) + b
            else: x = x.dot(w) + b
            if self.dropout != None and self.dropout > 0. and (i < len(self.layer_weights) - 1 or self.dropout_output):
                if mode == 'train': x = x * rng.binomial(size=x.shape, p=self.dropout, ndim=2)
                elif mode == 'test': x = x * self.dropout
                else: raise Exception('Invalid mode for MLP with dropout')
            for w1, w2 in self.res_weights[i + 1]: x += relu(x.dot(w1)).dot(w2)
        if self.output == 'linear': return x
        elif self.output == 'relu': return relu(x)
        elif self.output == 'softmax':
            if self.softmax_scale != 1.: x = x * self.softmax_scale
            return T.nnet.softmax(T.minimum(T.maximum(-60, x), 60))

    def _init_weight(self, n_in, n_out, scale=1.):
        return np.random.normal(scale=(scale * np.sqrt(4. / (n_in + n_out))), size=(n_in, n_out)).astype(theano.config.floatX)

    def _init_res_weight(self, n):
        return self._init_weight(n, n, scale=0.01)

    def get_params(self):
        weight_norm_params = self.layer_weight_gs if self.weight_norm else []
        return self.layer_weights + weight_norm_params + [aaa for a in self.res_weights for aa in a for aaa in aa] + self.layer_biases


def build_graph(n_timestep, n_channel, n_class, cfg):

    ## M1 (unsupervised stage)

    # Create neural networks
    m1_cnn_out_shape = (cfg['m1_cnn_h_depth'],) + tuple([np.int64(np.ceil(float(a) / cfg['m1_cnn_downscale'] ** (1 + cfg['m1_cnn_hidden_layers']))) for a in cfg['patch_shape']]) # Output of convolutional layers, per timestep
    m1_enc_cnn = CNN(np.prod(n_timestep * n_channel), n_timestep * m1_cnn_out_shape[0], downscale=cfg['m1_cnn_downscale'], hidden_layers=cfg['m1_cnn_hidden_layers'], h_depth=(n_timestep * cfg['m1_cnn_h_depth']), output='relu', res_depth=cfg['m1_cnn_res_depth'], name='m1_enc_cnn')
    m1_enc_mlp = MLP(n_timestep * np.prod(m1_cnn_out_shape), n_timestep * cfg['m1_n_z'], hidden_layers=cfg['m1_mlp_hidden_layers'], n_h=(n_timestep * cfg['m1_mlp_n_h']), res_depth=cfg['m1_mlp_res_depth'], name='m1_enc_mlp')
    m1_dec_mlps = tuple([MLP(cfg['m1_n_z'], np.prod(m1_cnn_out_shape), hidden_layers=cfg['m1_mlp_hidden_layers'], n_h=cfg['m1_mlp_n_h'], res_depth=cfg['m1_mlp_res_depth'], output='relu', name=('m1_dec_mlp_t%d' % i)) for i in range(n_timestep)])
    m1_dec_cnns = tuple([CNN(cfg['m1_cnn_h_depth'], n_channel, upscale=cfg['m1_cnn_downscale'], hidden_layers=cfg['m1_cnn_hidden_layers'], h_depth=cfg['m1_cnn_h_depth'], res_depth=cfg['m1_cnn_res_depth'], name=('m1_dec_cnn_t%d' % i)) for i in range(n_timestep)])

    # Create singleton parameters (log standard deviations)
    m1_q_z_ls_param, m1_rec_ls_param = [theano.shared(np.array(0, theano.config.floatX), name=name) for name in ('m1_q_z_ls_param', 'm1_rec_ls_param')]
    m1_q_z_ls, m1_rec_ls = cfg['m1_q_z_ls_scale'] * m1_q_z_ls_param, cfg['m1_rec_ls_scale'] * m1_rec_ls_param

    # Parse input
    m1_manifest = T.TensorType(theano.config.floatX, (False,) * 5)('m1_manifest') # (sample, time, band, row, column)
    m1_in = m1_manifest.reshape((-1, n_timestep * n_channel) + cfg['patch_shape']) # Collapse timesteps as parallel bands of one image

    # Feed input through encoder, both timesteps concatenated along the channel dimension
    m1_h = m1_enc_cnn(m1_in).flatten(2)
    m1_q_z_mean = m1_enc_mlp(m1_h)

    # Parse bottleneck activation
    m1_kld_loss = T.maximum(cfg['m1_n_z'] * cfg['m1_kld_cap'], kld_to_std_normal(m1_q_z_mean, m1_q_z_ls)).mean()
    m1_latent = m1_q_z_mean + m1_q_z_ls.exp() * rng.normal(m1_q_z_mean.shape) # Sample latent variable
    m1_latent_dc = [m1_latent[:, (t * cfg['m1_n_z']):((t + 1) * cfg['m1_n_z'])] for t in range(n_timestep)] # Decouple z here (reconstruct each timestep separately)

    # Feed bottleneck activation through decoder, each timestep separately
    m1_h_dc = [a(b).reshape((-1,) + m1_cnn_out_shape) for a, b in zip(m1_dec_mlps, m1_latent_dc)]

    # Parse output
    m1_rec_mean = T.stack([a(b) for a, b in zip(m1_dec_cnns, m1_h_dc)], 1) # Recouple reconstructions for each timestep at the output

    # Calculate loss from bottleneck activation and output
    m1_rec_loss = -normal_log_density(m1_manifest.flatten(3), m1_rec_mean.flatten(3), m1_rec_ls).mean()
    m1_loss = cfg['m1_kld_weight'] * m1_kld_loss + cfg['m1_rec_weight'] * m1_rec_loss

    # Sampled reconstruction from output (for debugging)
    m1_rec = m1_rec_mean + m1_rec_ls.exp() * rng.normal(m1_rec_mean.shape)

    # Create training and other functions
    m1_params = [aa for a in (m1_enc_cnn, m1_enc_mlp) + m1_dec_mlps + m1_dec_cnns for aa in a.get_params()] + [m1_q_z_ls_param, m1_rec_ls_param]
    m1_updates = adam(m1_loss, m1_params, learning_rate=cfg['m1_learning_rate'])
    m1_latent_fn = theano.function([m1_manifest], m1_latent)
    m1_metric_names = ('m1_q_z_ls', 'm1_q_z_mean', 'm1_rec_ls', 'm1_rec_mean', 'm1_kld_loss', 'm1_rec_loss', 'm1_loss')
    m1_train_fn = theano.function([m1_manifest], [m1_q_z_ls, m1_q_z_mean, m1_rec_ls, m1_rec_mean, m1_kld_loss, m1_rec_loss, m1_loss], updates=m1_updates)
    m1_rec_fn = theano.function([m1_latent], m1_rec)

    ## M2 (semi-supervised stage)

    # Create neural networks
    m2_enc_mlp = MLP(n_timestep * cfg['m1_n_z'], n_timestep * cfg['m2_n_h'], hidden_layers=(cfg['m2_hidden_layers'] - 1), n_h=(n_timestep * cfg['m2_n_h']), res_depth=cfg['m2_mlp_res_depth'], output='relu', dropout=cfg['m2_dropout'], dropout_output=True, name='m2_enc_mlp')
    m2_z_mlp = MLP(n_timestep * cfg['m2_n_h'], n_timestep * cfg['m2_n_z'], hidden_layers=1, n_h=(n_timestep * cfg['m2_n_h']), name='m2_z_mlp')
    m2_k_mlps = tuple([MLP(cfg['m2_n_h'], n_class, hidden_layers=1, n_h=cfg['m2_n_h'], output='softmax', softmax_scale=0.1, name=('m2_k_mlp_t%d' % i)) for i in range(n_timestep)])
    m2_dec_mlps = tuple([tuple([MLP(cfg['m2_n_z'], cfg['m1_n_z'], hidden_layers=cfg['m2_hidden_layers'], n_h=cfg['m2_n_h'], res_depth=cfg['m2_mlp_res_depth'], dropout=cfg['m2_dropout'], name=('m2_dec_mlp_t%d_k%d' % (i, ii))) for ii in range(n_class)]) for i in range(n_timestep)])

    # Create singleton parameters (log standard deviations, Markov chain parameters)
    m2_q_z_ls_param, m2_rec_ls_param = [theano.shared(np.array(0, theano.config.floatX), name=name) for name in ('m2_q_z_ls_param', 'm2_rec_ls_param')]
    m2_q_z_ls, m2_rec_ls = cfg['m2_q_z_ls_scale'] * m2_q_z_ls_param, cfg['m2_rec_ls_scale'] * m2_rec_ls_param
    m2_p_k0_params = theano.shared(np.zeros(n_class, theano.config.floatX), name='m2_q_z_k0_params')
    m2_p_tr_params = theano.shared(np.zeros((n_class, n_class), theano.config.floatX), name='m2_q_z_tr_params')
    m2_p_k0, m2_p_tr = [T.nnet.softmax(cfg['m2_p_k_scale'] * a) for a in (m2_p_k0_params.dimshuffle('x', 0), m2_p_tr_params)]
    m2_p_k = [m2_p_k0] + [None] * (n_timestep - 1)
    for i in range(n_timestep - 1): m2_p_k[i + 1] = m2_p_k[i].dot(m2_p_tr)
    m2_p_k = T.concatenate(m2_p_k)

    # Parse input
    m2_manifest = T.matrix('m2_manifest') # (batch, timestep * h)
    m2_labels = T.tensor3('labels') # (sample, time, class)
    m2_label_present = T.vector('label_present') # (sample)

    # Feed input through encoder
    m2_h_train, m2_h_test = [m2_enc_mlp(m2_manifest, rng=rng, mode=mode) for mode in ('train', 'test')]
    m2_q_z_mean_train, m2_q_z_mean_test = [m2_z_mlp(a) for a in (m2_h_train, m2_h_test)]
    m2_q_k_dc_train, m2_q_k_dc_test = [[aa(a[:, (t * cfg['m2_n_h']):((t + 1) * cfg['m2_n_h'])]) for t, aa in enumerate(m2_k_mlps)] for a in (m2_h_train, m2_h_test)] # Decouple hidden activation here to calculate q(k) separately for each timestep. (timestep, batch, class)

    # Parse bottleneck activation
    m2_q_k_train, m2_q_k_test = [T.stack(a, 1) for a in (m2_q_k_dc_train, m2_q_k_dc_test)] # Recouple q(k) for classifier output and for calculation of KLD. (batch, timestep, class)
    m2_z_kld = kld_to_std_normal(m2_q_z_mean_train, m2_q_z_ls)
    m2_k_kld = (m2_q_k_train.reshape((-1, n_timestep * n_class)) * (T.maximum(1e-9, m2_q_k_train.reshape((-1, n_timestep * n_class))).log() - T.maximum(1e-9, m2_p_k.flatten().dimshuffle('x', 0)).log())).sum(1)
    m2_kld_loss = T.maximum(cfg['m2_n_z'] * cfg['m2_kld_cap'], m2_z_kld + m2_k_kld).mean()
    m2_z = m2_q_z_mean_train + m2_q_z_ls.exp() * rng.normal(m2_q_z_mean_train.shape)
    m2_z_dc = [m2_z[:, (t * cfg['m2_n_z']):((t + 1) * cfg['m2_n_z'])] for t in range(n_timestep)] # Decouple z here (reconstruct each timestep separately). (timestep, batch, z)

    # Feed bottleneck activation through decoder
    m2_rec_mean_flat = T.stack([T.stack([aa(b, rng=rng, mode='train') for aa in a], 1) for a, b in zip(m2_dec_mlps, m2_z_dc)], 1).reshape((-1, n_class, cfg['m1_n_z'])) # Recouple class-conditional reconstructions for each timestep at the output. (batch * timestep, class, h)

    # Parse output
    m2_class_rec_flat = (m2_rec_mean_flat + m2_rec_ls.exp() * rng.normal(m2_rec_mean_flat.shape)).reshape((-1, n_class, cfg['m1_n_z']))
    m2_rec_sample_formatted = m2_manifest.reshape((-1, n_timestep, cfg['m1_n_z'])).dimshuffle(0, 1, 'x', 2)
    m2_rec_mean_formatted = m2_rec_mean_flat.reshape((-1, n_timestep, n_class, cfg['m1_n_z']))
    m2_rec_density = normal_log_density(m2_rec_sample_formatted, m2_rec_mean_formatted, m2_rec_ls)

    # Calculate loss from bottleneck activation and output
    m2_rec_loss = (m2_q_k_train * -m2_rec_density).sum((1, 2)).mean()
    m2_cls_loss = (m2_label_present.dimshuffle(0, 'x') * T.nnet.categorical_crossentropy(T.maximum(1e-9, m2_q_k_train.reshape((-1, n_class))), 0.8 * m2_labels.reshape((-1, n_class)) + 0.2 / n_class).reshape((-1, n_timestep))).sum() / m2_label_present.sum()
    m2_test_cls_loss = T.nnet.categorical_crossentropy(T.maximum(1e-9, m2_q_k_test.reshape((-1, n_class))), m2_labels.reshape((-1, n_class))).reshape((-1, n_timestep)).sum(1).mean()
    m2_loss = cfg['m2_kld_weight'] * m2_kld_loss + cfg['m2_rec_weight'] * m2_rec_loss + cfg['m2_cls_weight'] * m2_cls_loss

    # Sampled reconstruction from output (for debugging)
    m2_q_k_train_flat = m2_q_k_train.transpose(0, 2, 1).reshape((-1, n_class)) # (batch * timestep, class)    
    m2_q_k_less_than = rng.uniform(size=m2_q_k_train_flat.shape[0].dimshuffle('x')).dimshuffle(0, 'x') < T.cumsum(m2_q_k_train_flat, 1)
    m2_k_sample_flat = (T.concatenate((m2_q_k_less_than, T.zeros((m2_q_k_less_than.shape[0], 1))), 1) * (1 - T.concatenate((T.zeros((m2_q_k_less_than.shape[0], 1)), m2_q_k_less_than), 1)))[:, :-1]
    m2_rec = (m2_class_rec_flat * m2_k_sample_flat.dimshuffle(0, 1, 'x')).sum(1).reshape((-1, n_timestep * cfg['m1_n_z'])) # (batch, timestep * h)

    # Create training and other functions
    m2_params = [aa for a in (m2_enc_mlp, m2_z_mlp) + m2_k_mlps + tuple([aa for a in m2_dec_mlps for aa in a]) for aa in a.get_params()] + [m2_q_z_ls_param, m2_rec_ls_param, m2_p_k0_params, m2_p_tr_params]
    m2_updates = adam(m2_loss, m2_params, learning_rate=cfg['m2_learning_rate'])
    m2_metrics = [m2_q_z_ls, m2_q_z_mean_train, m2_q_k_train, m2_p_k, m2_rec_ls, m2_rec_mean_flat, m2_kld_loss, m2_rec_loss, m2_cls_loss, m2_loss]
    m2_metric_names = ('m2_q_z_ls', 'm2_q_z_mean', 'm2_q_k', 'm2_p_k', 'm2_rec_ls', 'm2_rec_mean_flat', 'm2_kld_loss', 'm2_rec_loss', 'm2_cls_loss', 'm2_loss')
    m2_train_fn = theano.function([m2_manifest, m2_labels, m2_label_present], m2_metrics, updates=m2_updates)
    m2_eval_fn = theano.function([m2_manifest, m2_labels], m2_test_cls_loss)
    m2_cls_fn = theano.function([m2_manifest], m2_q_k_test)
    m2_compress_rec_fn = theano.function([m2_manifest], m2_rec)

    metric_names = {'m1_train': m1_metric_names, 'm2_train': m2_metric_names}
    network_functions = {'m1_train': m1_train_fn, 'm1_latent': m1_latent_fn, 'm1_rec': m1_rec_fn, 'm2_train': m2_train_fn, 'm2_eval': m2_eval_fn, 'm2_cls': m2_cls_fn, 'm2_compress_rec': m2_compress_rec_fn}
    return metric_names, network_functions
