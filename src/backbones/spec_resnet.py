import torch.nn as nn
import timm
from .spectral_adapter import SpectralAdapter



class CustomBottleneck(nn.Module):
    """Custom bottleneck block for ResNet architecture.
    
    This implements a modified bottleneck residual block with three convolutional layers
    and a skip connection. It follows the standard ResNet bottleneck design but allows
    for customization of channels and stride.
    """
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        """Initialize the CustomBottleneck block.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride value for the middle convolutional layer.
            downsample (nn.Module, optional): Downsampling layer for the residual connection
                when spatial dimensions change. Default: None.
        """
        super(CustomBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        """Forward pass through the bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the bottleneck block.
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class CustomResNet50(nn.Module):
    """Custom ResNet50 implementation that accepts features from a spectral adapter.
    
    This model modifies the standard ResNet50 architecture to work with features 
    from the spectral adapter rather than direct RGB inputs. It allows for dilation
    of convolutional layers to increase receptive field and preserve spatial resolution.
    """
    def __init__(self, num_classes=1000, replace_stride_with_dilation=[False, False, False, False], return_features=False):
        """Initialize the CustomResNet50 model.
        
        Args:
            num_classes (int): Number of classes for the classification head. Default: 1000.
            replace_stride_with_dilation (list): List of booleans indicating whether to replace 
                stride with dilation in each ResNet stage. Default: [False, False, False, False].
            return_features (bool): If True, return features before the global pooling layer.
                Default: False.
        """
        super(CustomResNet50, self).__init__()

        base_model = timm.create_model('resnet50', pretrained=False)
        self.return_features = return_features
        self.num_features = base_model.num_features

        downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.layer1 = nn.Sequential(
            CustomBottleneck(128, 256, stride=2, downsample=downsample),
            base_model.layer1[1],
            base_model.layer1[2]
        )

        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        dilation_value = 1  # Initial dilation value

        # Apply dilation if specified
        for layer, replace_dilation in zip(layers, replace_stride_with_dilation):
            if replace_dilation:
                for block in layer:
                    if block.downsample is not None:
                        block.downsample[0].stride = (1, 1)
                    block.conv2.stride = (1, 1)
                    block.conv2.dilation = (dilation_value, dilation_value)
                    block.conv2.padding = (dilation_value, dilation_value)
                dilation_value *= 2  # Double the dilation for the next layer


        self.global_pool = base_model.global_pool
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        """Forward pass through the CustomResNet50.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 128, height, width].
                Expected to be the output from a spectral adapter.
                
        Returns:
            torch.Tensor: Either feature maps or classification logits based on return_features.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.return_features:
            return x
        
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x

    

class SpecResNet50(nn.Module):
    """Spectral ResNet50 model combining a spectral adapter with a custom ResNet50.
    
    This model is designed for hyperspectral image classification. It first processes
    the spectral bands using a SpectralAdapter, then feeds the resulting features
    into a modified ResNet50 architecture for spatial feature extraction and classification.
    """
    def __init__(self, num_classes, replace_stride_with_dilation=[False, False, False, False], return_features=False):
        """Initialize the Spectral ResNet50 model.
        
        Args:
            num_classes (int): Number of output classes.
            replace_stride_with_dilation (list): List of booleans indicating whether to 
                replace stride with dilation in each ResNet stage. Default: [False, False, False, False].
            return_features (bool): If True, return features before the global pooling layer.
                Default: False.
        """
        super(SpecResNet50, self).__init__()
        
        self.spectral_adapter = SpectralAdapter()
        
        self.resnet = CustomResNet50(num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation, return_features=return_features)
        self.num_features = self.resnet.num_features

    def forward(self, x):
        """Forward pass through the Spectral ResNet50.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, depth, height, width]
                where depth is the spectral dimension.
                
        Returns:
            torch.Tensor: Output tensor with classification logits or features.
        """
        x = self.spectral_adapter(x)
        return self.resnet(x)

    
    def get_classifier(self):
        """Get the classification head of the model.
        
        Returns:
            nn.Module: The final fully connected layer used for classification.
        """
        return self.resnet.fc



class ModifiedResNet50(nn.Module):
    """ Modified ResNet50 with optional dilation and flexible input channels. """
    def __init__(self, num_classes=1000, in_channels=3, pretrained=False, replace_stride_with_dilation=None, return_features=False):
        """Initialize the ModifiedResNet50 model.
        
        Args:
            num_classes (int): Number of output classes. Default: 1000.
            in_channels (int): Number of input channels. Default: 3.
            pretrained (bool): Whether to initialize with pretrained weights. Default: False.
            replace_stride_with_dilation (bool): If True, replace stride with dilation to 
                increase receptive field while maintaining spatial resolution. Default: None.
            return_features (bool): If True, return features before the global pooling layer.
                Default: False.
        """
        super(ModifiedResNet50, self).__init__()
        # Create base ResNet50 model from timm
        base_model = timm.create_model('resnet50', pretrained=pretrained, in_chans=in_channels)
        self.return_features = return_features
        self.num_features = base_model.num_features

        if replace_stride_with_dilation:
            # Modify stem: remove stride from initial convolution and pooling
            base_model.conv1.stride = (1, 1)
            base_model.maxpool = nn.Identity()

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.act1 = base_model.act1  # Activation layer name in timm
        self.maxpool = base_model.maxpool

        # Use the layers from the base model
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Apply dilation to the layers
        if replace_stride_with_dilation:
            dilation_value = 2

            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                if replace_stride_with_dilation:
                    for block in layer:
                        if block.downsample is not None:
                            block.downsample[0].stride = (1, 1)
                        block.conv2.stride = (1, 1)
                        block.conv2.dilation = (dilation_value, dilation_value)
                        block.conv2.padding = (dilation_value, dilation_value)
                    dilation_value *= 2

        self.global_pool = base_model.global_pool
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(base_model.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ModifiedResNet50.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
                
        Returns:
            torch.Tensor: Output tensor with classification logits or features.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.return_features:
            return x

        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    def get_classifier(self):
        """Get the classification head of the model.
        
        Returns:
            nn.Module: The final fully connected layer used for classification.
        """
        return self.fc