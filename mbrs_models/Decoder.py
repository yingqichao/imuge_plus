from . import *


# class Decoder(nn.Module):
# 	'''
# 	Decode the encoded image and get message
# 	'''
#
# 	def __init__(self, H, W, message_length, blocks=4, channels=64):
# 		super(Decoder, self).__init__()
#
# 		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
# 		keep_blocks = max(blocks - stride_blocks, 0)
#
# 		self.first_layers = nn.Sequential(
# 			ConvBNRelu(3, channels),
# 			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
# 			ConvBNRelu(channels * (2 ** stride_blocks), channels),
# 		)
# 		self.keep_layers = SENet(channels, channels, blocks=keep_blocks)
#
# 		self.final_layer = ConvBNRelu(channels, 1)
#
# 	def forward(self, noised_image):
# 		x = self.first_layers(noised_image)
# 		x = self.keep_layers(x)
# 		x = self.final_layer(x)
# 		x = x.view(x.shape[0], -1)
# 		return x

## Define the NN architecture
class Decoder_MLP(nn.Module):
    def __init__(self,in_neurons=16*16*3, hidden=512, out_neurons=2):
        super(Decoder_MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_neurons, hidden),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_neurons)
        )

    def forward(self, x):
        # flatten image input
        # x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, out_num=4):

        super(Decoder, self).__init__()
        self.channels = 64

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(8):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, out_num))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(out_num, out_num)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = (torch.tanh(self.linear(x))+1)/2
        return x


class Decoder_Diffusion(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Decoder_Diffusion, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(diffusion_length))))

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(self.diffusion_length ** 0.5)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet(channels, channels, blocks=1)

		self.final_layer = ConvBNRelu(channels, 1)

		self.message_layer = nn.Linear(self.diffusion_length, message_length)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)

		x = self.message_layer(x)
		return x
