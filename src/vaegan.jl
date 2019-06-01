using Flux, Flux.Data.MNIST, Statistics
using Flux: throttle, params, binarycrossentropy, crossentropy
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview

include("data_loader.jl")
# imgs = MNIST.images()

BATCH_SIZE = 512
REAL_LABEL =  ones(1, BATCH_SIZE)
FAKE_LABEL = zeros(1, BATCH_SIZE)
#
# data = [reshape(hcat(Array(channelview.(imgs))...), 28, 28, 1,:) for imgs in partition(imgs, BATCH_SIZE)]
# data = gpu.(data)

data = load_dataset_as_batches("C:/Users/manju/Downloads/celeba-dataset/img_align_celeba/img_align_celeba/", BATCH_SIZE)
data = gpu.(data)

println(size(data))

NUM_EPOCHS = 50
training_steps = 0
GAMMA = 25
BETA = 5

discriminator_eta = 0.0001f0
generator_eta = 0.0001f0

encoder_features = Chain(Conv((5,5), 3 => 32,leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(32),
	Conv((5, 5), 32 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(64),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	x -> reshape(x, :, size(x, 4)),
	Dense(4096, 2048),
	x -> relu.(x))

encoder_mean = Chain(encoder_features, Dense(1024*2, 512))

encoder_logsigma = Chain(encoder_features, Dense(1024*2, 512), x -> tanh.(x))

encoder_latent(x) = encoder_mean(x) + randn(512, 512) * exp.(encoder_logsigma(x)./2)

decoder_generator = Chain(Dense(512, 32*8*4*4),
	x -> reshape(x, 4, 4, 256, :),
	BatchNorm(256),
	x -> relu.(x),
	ConvTranspose((4, 4), 256 => 128, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(128),
	ConvTranspose((4, 4), 128 => 64, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(64),
	ConvTranspose((4, 4), 64 => 32, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(32),
	ConvTranspose((4, 4), 32 => 3, relu, stride = (2, 2), pad = (1, 1)),
	x -> tanh.(x))

discriminator_featuremap = Chain(Conv((5, 5), 3 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	Conv((5, 5), 256 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)))
	# BatchNorm(256),
	# x -> reshape(x, :, size(x, 4)))

discriminator = Chain(discriminator_featuremap, BatchNorm(256), x -> reshape(x, :, size(x, 4)),
	Dense(1024*4, 1), x -> sigmoid.(x))

discriminator_similar = Chain(discriminator_featuremap, BatchNorm(256), x -> reshape(x, :, size(x, 4)), Dense(1024*4, 1))

function auxiliary_Z(latent_vector)
	return abs.(randn(size(latent_vector)...))
end

opt_encoder = ADAM(0.0003, (0.9, 0.999))
function prior_loss(latent_vector, auxiliary_Z)
	entropy = sum(latent_vector .* log.(latent_vector)) *1 //size(latent_vector,2)
 	cross_entropy = crossentropy(auxiliary_Z, latent_vector)
 	return entropy + cross_entropy
end

function discriminator_loss(reconstructed_data, sample_data, real_data, REAL_LABEL, FAKE_LABEL)
	reconstruction_loss = binarycrossentropy(discriminator(reconstructed_data), FAKE_LABEL)
	sampling_loss = binarycrossentropy(discriminator(sample_data), FAKE_LABEL)
	real_loss = binarycrossentropy(discriminator(real_data), REAL_LABEL)
	return reconstruction_loss + sampling_loss + real_loss
end


latent_vector = encoder_latent(X)
log_sigma = encoder_logsigma(X)
enc_mean = encoder_mean(X)
Z_prior = auxiliary_Z(latent_vector)
X_reconstructed = decoder_generator(latent_vector)
X_p = decoder_generator(Z_prior)

x1 = discriminator(X_reconstructed)
xp = discriminator(X_p)
x_real = discriminator(X)

x_sim = discriminator_similar(X_reconstructed)
x_sim_real = discriminator_similar(X)

reconstruction_loss = Flux.mse(x_sim, x_sim_real)
decoder_loss = GAMMA * reconstruction_loss - discriminator_loss(X_reconstructed, X_p, X, REAL_LABEL, FAKE_LABEL)

encoder_loss = -0.5*(1 .+ log_sigma .- (enc_mean .* enc_mean) .- exp.(log_sigma))/ (BATCH_SIZE*784) + BETA * reconstruction_loss
