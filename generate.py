import matplotlib as mpl
mpl.use('TkAgg')

import os
import sys
import argparse

from griffin_lim import *
from model_iaf import *
from util import *

import librosa
import librosa.display


import matplotlib.pyplot as plt

param = {
    "N_FFT": 1024,
    "HOP_LENGTH": 256,
    "SAMPLING_RATE": 16000,
    "MELSPEC_BANDS": 128,
    "sample_secs": 2
}

N_FFT = param['N_FFT']
HOP_LENGTH = param['HOP_LENGTH']
SAMPLING_RATE = param['SAMPLING_RATE']
MELSPEC_BANDS = param['MELSPEC_BANDS']
sample_secs = param['sample_secs']
num_samples_dataset = int(sample_secs * SAMPLING_RATE)

logdir = './logdir'
max_checkpoints = 5
num_steps = 10000
checkpoint_every = 500
batch_size = 64
learning_rate = 1e-3
beta=1.0
model_params = 'params.json'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Spectrogram VAE')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. ')
    parser.add_argument('--file_in', type=str, nargs='*',
                        help='Input file(s) from which to generate new audio. If none, sample random point in latent space')
    parser.add_argument('--file_out', type=str, default='generated',
                        help='Output file for storing new audio. ')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Output directory for storing new audio. ')
    parser.add_argument('--num_outputs', type=int, default=1,
                        help='number of outputs')
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def main():

    args = get_arguments()
    
    batch_size = 1

    # Look for original parameters
    print('Loading existing parameters.')
    print(f'{args.logdir}/params.json')
    with open(f'{args.logdir}/params.json', 'r') as f:
        param = json.load(f)

    # Set correct batch size in deconvolution shapes
    deconv_shape = param['deconv_shape']
    for k, s in enumerate(deconv_shape):
        actual_shape = s
        actual_shape[0] = batch_size
        deconv_shape[k] = actual_shape
    param['deconv_shape'] = deconv_shape

    # Create network.
    net = VAEModel(param,
                   batch_size)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)

    try:
        saved_global_step = load(saver, sess, args.logdir)

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    # Check if directory for saving exists
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(args.num_outputs):
        embedding_mean_batch = np.float32(np.random.standard_normal((1, param['dim_latent'])))

        # Decode the mean embedding
        out_mean = net.decode(embedding_mean_batch)
        output_mean = sess.run(out_mean)

        spec_out = (np.squeeze(output_mean[0])-1.0)*80.0

        # Reconstruct audio
        audio = griffin_lim(spec_out)
        audio_file = f'{out_dir}/{args.file_out}_{i}.wav'
        librosa.output.write_wav(audio_file, audio / np.max(audio), sr=SAMPLING_RATE)


if __name__ == '__main__':
    main()