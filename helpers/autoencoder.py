from .batch_iterator import iterate_minibatches
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
import pickle as pkl

def build_deep_autoencoder(img_shape, code_size=32):
    class _encoder:
        l_inp = InputLayer((None,) + img_shape)
        input_images = l_inp.input_var
        l_inp = BatchNormLayer(l_inp)
        l_shaped = DimshuffleLayer(l_inp, (0,3,1,2))
        l_conv = Conv2DLayer(l_shaped, code_size, 3, nonlinearity=T.nnet.elu, pad=1)
        l_pool = MaxPool2DLayer(l_conv,2)
        #l_pool = BatchNormLayer(l_pool)
        l_pool = DropoutLayer(l_pool, 0.2)
        l_conv = Conv2DLayer(l_pool, code_size, 3, nonlinearity=T.nnet.elu, pad=1)
        l_pool = MaxPool2DLayer(l_conv,2)
        #l_pool = BatchNormLayer(l_pool)
        l_pool = DropoutLayer(l_pool, 0.1)

        l_conv = Conv2DLayer(l_pool, code_size, 3, nonlinearity=T.nnet.elu, pad=1)
        l_pool = MaxPool2DLayer(l_conv,2)
        l_pool = BatchNormLayer(l_pool)
        l_pool = DropoutLayer(l_pool, 0.2)
        l_flat = flatten(l_pool)
        l_code = DenseLayer(l_flat, code_size, nonlinearity=T.tanh)
        l_code = BatchNormLayer(l_code)
        l_code = DropoutLayer(l_code, 0.2)
        l_code = DenseLayer(l_code, code_size, nonlinearity=None)

        code = get_output(l_code)
        det_code = get_output(l_code, deterministic=True)

        weights = get_all_params(l_code, trainable= True)
        l2 = lasagne.regularization.regularize_network_params(l_code, lasagne.regularization.l2)
        l1 = lasagne.regularization.regularize_network_params(l_code, lasagne.regularization.l1)

        encode = theano.function([input_images], det_code)

    class _decoder:
        l_inp = InputLayer((None,code_size))
        l_flat = DenseLayer(l_inp, code_size * 3, nonlinearity= T.tanh)
        #l_flat = BatchNormLayer(l_flat)
        l_flat = DropoutLayer(l_flat, 0.2)
        l_flat = DenseLayer(l_inp, code_size * 18, nonlinearity= T.tanh)
        #l_flat = BatchNormLayer(l_flat)
        l_flat = DropoutLayer(l_flat, 0.2)
        l_unflatten = ReshapeLayer(l_flat,(-1,2 * code_size,3,3))

        l_deconv = Deconv2DLayer(l_unflatten, code_size, 3, crop = 1)
        while(get_output_shape(l_deconv)[-2] < img_shape[-3] or get_output_shape(l_deconv)[-1] < img_shape[-2]):
            l_upscale = Upscale2DLayer(l_deconv, 2)
            #l_upscale = BatchNormLayer(l_upscale)
            l_upscale = DropoutLayer(l_upscale, 0.2)
            l_deconv = Deconv2DLayer(l_upscale, code_size, 3, crop = 2)
        _, _, w,h = get_output_shape(l_deconv)

        """ #too slow
        def conv_params(w, target):
            w_s = w // target
            w_f = w % w_s
            w_p = target - (w - w_f) // w_s - 1
            while w_p < 0:
                w_f += w_s
                w_p = target - (w - w_f) // w_s - 1
            print(w,target,w_f, w_s, w_p)
            return w_f, w_s, w_p

        l_reconstructed = Conv2DLayer(l_deconv, img_shape[-1], *zip(conv_params(w, img_shape[-3]),conv_params(h, img_shape[-2])))
        """
        def pool_params(w, target):
            w_s = (w // target)
            w_p = - (w % (w_s * target))
            return w_s, w_p // 2
        sw, pw = pool_params(w, img_shape[-3])
        sh, ph = pool_params(h, img_shape[-2])
        l_reconstructed = Pool2DLayer(l_deconv, pool_size = (sw, sh), pad=(pw,ph)) #crop
        l_reconstructed = Conv2DLayer(l_reconstructed, img_shape[-1],1)

        l_reconstructed = DimshuffleLayer(l_reconstructed, (0, 2,3,1))

        reconstructed = get_output(l_reconstructed)
        det_reconstructed = get_output(l_reconstructed, deterministic = True)

        weights = get_all_params(l_reconstructed,trainable= True)
        l2 = lasagne.regularization.regularize_network_params(l_reconstructed, lasagne.regularization.l2)
        l1 = lasagne.regularization.regularize_network_params(l_reconstructed, lasagne.regularization.l1)

        decode = theano.function([l_inp.input_var], det_reconstructed)
    class autoencoder:
        encoder = _encoder
        decoder = _decoder
        # regularization params
        encoder_l2 = theano.shared(0.0,"encoder_l2")
        decoder_l2 = theano.shared(0.0,"decoder_l2")
        # apply
        l_inp = encoder.l_inp
        code = encoder.code
        reconstructed = get_output(decoder.l_reconstructed, {decoder.l_inp: code})
        #fit
        all_weights = encoder.weights + decoder.weights
        regularization = encoder_l2 * encoder.l2 + decoder_l2 * decoder.l2

        loss = lasagne.objectives.squared_error(encoder.input_images, reconstructed).mean() + regularization
        updates = lasagne.updates.adam(loss, all_weights)

        _fit = theano.function([encoder.input_images], loss, updates= updates, allow_input_downcast= True)
        predict = theano.function([encoder.input_images], reconstructed)

        _encode = encoder.encode
        _decode = decoder.decode

        @staticmethod
        def encode(X, batch_size=64):
            res = []
            for batch in iterate_minibatches(X, batchsize=batch_size, shuffle=False):
                res.append(autoencoder._encode(batch))
            return np.concatenate(res)

        @staticmethod
        def decode(X, batch_size=64):
            res = []
            for batch in iterate_minibatches(X, batchsize=batch_size, shuffle=False):
                res.append(autoencoder._decode(batch))
            return np.concatenate(res)

        @staticmethod
        def fit(X, batch_size= 32, n_epochs= 1, epochal= None):

            def _epochal(epoch, losses):
                pass
            epochal = epochal or _epochal
            hist = []
            n_batches = np.ceil(X.shape[0] / float(batch_size))

            for epoch in range(n_epochs):
                loss = []

                t = tqdm_notebook(iterate_minibatches(X, batchsize= batch_size, shuffle= True),
                                  total= n_batches)

                for x in t:
                    loss.append(autoencoder._fit(x))
                    t.set_description('Epoch {0}, mean loss: {1}'.format(epoch, np.mean(loss)))
                hist.append(np.mean(loss))
                t.set_description('End of Epoch {0}, mean loss: {1}'.format(epoch, hist[-1]))
                epochal(epoch, hist)

    return autoencoder


def load(deep, fname):
    with open(fname, 'rb') as f:
        for w in get_all_params(deep.encoder.l_code) + get_all_params(deep.decoder.l_reconstructed):
            #if w.name in {'beta', 'gamma'}:
            #    continue
            w.set_value(pkl.load(f))

def dump(deep, fname):
    with open(fname, 'wb') as f:
        for w in get_all_params(deep.encoder.l_code) + get_all_params(deep.decoder.l_reconstructed):
            #if w.name in {'beta', 'gamma'}:
            #    continue
            pkl.dump(w.get_value(), f)

def load_autoencoder(ing_shape,weights_file,code_size = 64):
    deep = build_deep_autoencoder(ing_shape,code_size)
    load(deep, weights_file)
    return deep
