using Flux, Flux.Data.MNIST, Statistics
using Flux: throttle, params
using Distributions
import Distributions: logpdf
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview

imgs = MNIST.images()

BATCH_SIZE = 128

data = [reshape(hcat(Array(channelview.(imgs))...), 28, 28, 1,:) for imgs in partition(imgs, BATCH_SIZE)]
data = gpu.(data)


NUM_EPOCHS = 20
channels = 128
hidden_dim = 7 * 7 * channels
training_steps = 0
verbose_freq = 100

discriminator_eta = 0.0001f0
generator_eta = 0.0001f0

encoder_features = Chain(Conv((5,5), 1 => 32,leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(32),
	Conv((5, 5), 32 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(64),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	x -> reshape(x, :, size(x, 4)))

encoder_mean = Chain(encoder_features, Dense(1024, 512))

encoder_logsigma = Chain(encoder_features, Dense(1024, 512), x -> tanh.(x))



decoder_generator = Chain(Dense(512, 32*8*28*28), 
	x -> reshape(x, 28, 28, 256, :),
	BatchNorm(256),
	x -> relu.(x),
	ConvTranspose((5, 5), 256 => 128, relu; stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	ConvTranspose((5, 5), 128 => 64, relu; stride = (2, 2), pad = (2, 2)),
	BatchNorm(64),
	ConvTranspose((5, 5), 64 => 32, relu; stride = (2, 2), pad = (2, 2)),
	BatchNorm(32),
	ConvTranspose((5, 5), 32 => 1, relu; stride = (2, 2), pad = (2, 2)),
	x -> tanh.(x))

discriminator_featuremap = Chain(Conv((5, 5), 1 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	Conv((5, 5), 256 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)))
	# BatchNorm(256),
	# x -> reshape(x, :, size(x, 4)))

decoder = Chain(discriminator_featuremap, BatchNorm(256), x -> reshape(x, :, size(x, 4)),
	Dense(1024, 1), x -> sigmoid.(x))

opt_encoder = ADAM(0.0003, (0.9, 0.999))





	


	



