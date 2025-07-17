import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 定义Dual_Branch_Feature_Fusion_Block模块
class Dual_Branch_Feature_Fusion_Block(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Dual_Branch_Feature_Fusion_Block, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


# 定义MFAB模块
def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


class Modal_Fusion_Attention_Block(nn.Module):
    """Fuse two features into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None, img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1
        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse


# 定义CrossAttention模块
class Cross_Modal_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_Modal_Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):
        batch_size, length, channels = x1.size()

        # Linear projection to obtain queries, keys, and values
        qkv1 = self.qkv_proj(x1).view(batch_size, length, self.num_heads, 3, channels // self.num_heads).permute(3, 0,
                                                                                                                 2, 1,
                                                                                                                 4)
        qkv2 = self.qkv_proj(x2).view(batch_size, length, self.num_heads, 3, channels // self.num_heads).permute(3, 0,
                                                                                                                 2, 1,
                                                                                                                 4)

        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        # Cross-attention calculation
        attn_weights1 = (q1 @ k2.transpose(-2, -1)) * (1.0 / (k1.size(-1) ** 0.5))
        attn_weights2 = (q2 @ k1.transpose(-2, -1)) * (1.0 / (k2.size(-1) ** 0.5))

        attn_weights1 = F.softmax(attn_weights1, dim=-1)
        attn_weights2 = F.softmax(attn_weights2, dim=-1)

        out1 = (attn_weights1 @ v2).transpose(1, 2).contiguous().view(batch_size, length, channels)
        out2 = (attn_weights2 @ v1).transpose(1, 2).contiguous().view(batch_size, length, channels)

        # Output projection
        out1 = self.out_proj(out1)
        out2 = self.out_proj(out2)

        return out1, out2


# 定义Adaptive_Frequency_Filtering_Block模块
class Adaptive_Frequency_Filtering_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, N, C = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across N dimension and then compute median
        flat_energy = energy.view(B, -1)  # Flattening
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((
                                     normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the N dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.training:  # Use adaptive filter only during training
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


# 定义最终的整合模型
class MultimodalClassificationModel(nn.Module):
    def __init__(self, num_classes=9):
        super(MultimodalClassificationModel, self).__init__()
        # Ultrasound branch
        self.us_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.adaptive_spectral_block = Adaptive_Frequency_Filtering_Block(dim=64)

        # Image branch
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.Dual_Branch_Fusion = Dual_Branch_Feature_Fusion_Block(dim=64, num_heads=4, bias=True)

        # Cross Attention module
        self.cross_attention = Cross_Modal_Attention(embed_dim=64, num_heads=4)

        # TFAM module
        self.tfam = Modal_Fusion_Attention_Block(in_channel=64)

        # Classification layer
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, us_input, img_input):
        # us_input: [batchsize, length, 1]
        # img_input: [batchsize, 3, 224, 224]

        # Ultrasound branch
        us_input = us_input.permute(0, 2, 1)
        us_feat = self.us_conv(us_input)
        us_feat = us_feat.permute(0, 2, 1)
        us_feat = self.adaptive_spectral_block(us_feat)

        # Image branch
        img_feat = self.img_conv(img_input)
        img_feat = self.Dual_Branch_Fusion(img_feat)

        # Flatten spatial dimensions
        img_feat = img_feat.view(img_feat.size(0), img_feat.size(1), -1)
        img_feat = img_feat.permute(0, 2, 1)

        # Cross Attention
        # Reduce features to same length
        target_sequence_length = 1024  # 新的序列长度，确保可以重塑为二维形状
        us_feat_reduced = F.adaptive_avg_pool1d(us_feat.permute(0, 2, 1), output_size=target_sequence_length).permute(0,
                                                                                                                      2,
                                                                                                                      1)  # [batchsize,1024,64]
        img_feat_reduced = F.adaptive_avg_pool1d(img_feat.permute(0, 2, 1), output_size=target_sequence_length).permute(
            0, 2, 1)  # [batchsize,1024,64]

        cross_feat_us, cross_feat_img = self.cross_attention(us_feat_reduced,
                                                             img_feat_reduced)  # Each is [batchsize, 1024, 64]

        # Combine features using TFAM
        # Reshape to [batchsize, channels, height, width]
        spatial_size = int(math.sqrt(target_sequence_length))  # 32
        cross_feat_us = cross_feat_us.permute(0, 2, 1).view(-1, 64, spatial_size, spatial_size)
        cross_feat_img = cross_feat_img.permute(0, 2, 1).view(-1, 64, spatial_size, spatial_size)

        fused_feat = self.tfam(cross_feat_us, cross_feat_img)

        # Global Average Pooling
        fused_feat = F.adaptive_avg_pool2d(fused_feat, output_size=1)
        fused_feat = fused_feat.view(fused_feat.size(0), -1)

        # Classification
        logits = self.classifier(fused_feat)  # [batchsize, num_classes]

        # Output softmax probabilities
        probs = F.softmax(logits, dim=1)

        return probs
# 测试示例



if __name__ == "__main__":
    model = MultimodalClassificationModel(num_classes=9)
    us_input = torch.randn(32, 10501, 1)
    img_input = torch.randn(32, 3, 224, 224)

    output = model(us_input, img_input)
    print("Output shape:", output.shape)
    print("Output:", output)

