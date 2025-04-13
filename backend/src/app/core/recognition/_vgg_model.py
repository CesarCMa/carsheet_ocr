from torch import nn


class VGGModel(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        """
        A CRNN (Convolutional Recurrent Neural Network) model for optical character recognition (OCR).

        The network architecture consists of three main components:
        1. Convolutional layers: Automatically extract a feature sequence from each input image.
        2. Recurrent layers: Make predictions for each frame of the feature sequence outputted by the convolutional layers.
        3. Transcription layer: Translate the per-frame predictions by the recurrent layers into a label sequence.

        Args:
            input_channel (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            output_channel (int): Number of output channels for the feature extractor.
            hidden_size (int): Number of hidden units in the recurrent layers.
            num_class (int): Number of output classes for the final prediction.
        """
        super(VGGModel, self).__init__()
        # Feature extraction stage
        self.FeatureExtraction = _VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # Sequence modeling with bidirectional LSTM
        self.SequenceModeling = nn.Sequential(
            _BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            _BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size

        # Prediction stage
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self.SequenceModeling(visual_feature)

        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


class _VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(_VGG_FeatureExtractor, self).__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.ConvNet(input)


class _BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(_BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
