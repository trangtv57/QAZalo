import torch
import torch.nn as nn
from torch.nn import functional as F


class CNNFeatureExtract2D(nn.Module):
    def __init__(self, embedding_dim, filter_num,
                 kernel_sizes=[2, 3],
                 dropout_cnn=0.5):

        super(CNNFeatureExtract2D, self).__init__()
        self.embedding_dim = embedding_dim
        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1,
                                              self.filter_num,
                                              (kernel_size, self.embedding_dim)) for kernel_size in self.kernel_sizes])

        self.dropout_layer = nn.Dropout(dropout_cnn)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        output_relu = F.relu(conv_out).squeeze(3)
        output_maxpool1d = F.max_pool1d(output_relu, output_relu.size(2)).squeeze(2)
        return output_maxpool1d

    def forward(self, embedding_feature):
        x = embedding_feature.unsqueeze(1)
        # embedding feature has shape (batch_size, num_seq, embedding_length)
        list_output_cnn = [self.dropout_layer(self.conv_block(x, conv_layer)) for conv_layer in self.convs]
        output_feature = torch.cat(list_output_cnn, 1)
        return output_feature


class CNNFeatureExtract1D(nn.Module):
    def __init__(self, char_embedding_dim, windows_size, dropout_cnn):
        super(CNNFeatureExtract1D, self).__init__()

        self.char_embedding_dim = char_embedding_dim
        self.windows_size = windows_size

        self.conv1ds = nn.ModuleList([nn.Conv1d(in_channels=self.char_embedding_dim,
                                                out_channels=self.char_embedding_dim,
                                                kernel_size=k) for k in windows_size
                                      ])

        self.dropout_cnn = nn.Dropout(dropout_cnn)

    def forward(self, char_embedding_feature):
        # char_embedding_feature: shape: (batch_size, max_len character of word, embedding dim)
        char_embedding_feature = char_embedding_feature.permute(0, 2, 1)

        conved = [self.dropout_cnn(F.relu(conv(char_embedding_feature))) for conv in self.conv1ds]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout_cnn(torch.cat(pooled, dim=1))
        return cat


if __name__ == '__main__':
    a = torch.rand((3, 5, 32))
    print(a.shape)
    test_cnn = CNNFeatureExtract1D(32, [2, 3], 0.1)
    out_cnn = test_cnn(a)

    print(out_cnn.shape)
