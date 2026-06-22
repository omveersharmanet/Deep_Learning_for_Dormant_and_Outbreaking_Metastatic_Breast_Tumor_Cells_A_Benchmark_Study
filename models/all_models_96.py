import torch
import torch.nn as nn
import torchvision.models as models
from segmentation_models_pytorch import Unet, UnetPlusPlus
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import ViTModel, ViTConfig, SwinModel, SwinConfig
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import ResNet34_Weights,ResNet50_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import DenseNet121_Weights,DenseNet169_Weights,DenseNet201_Weights
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import (
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights
)
from transformers import ViTModel, ViTConfig
import timm

import torch.nn.functional as F

from cellpose.models import CellposeModel
from segmentation_models_pytorch import Unet # Assuming SMP library
from segmentation_models_pytorch import UnetPlusPlus
from cellpose.models import CellposeModel
import segmentation_models_pytorch as smp
from transformers import SegformerModel, SegformerConfig
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
# -------------------------
# 1. CNN2D
# -------------------------
class CNN2D(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.num_layers = args.num_layers
        self.channels = args.channels
        self.dropout_rate = getattr(args, "dropout", 0.5)
        in_ch = args.sequence_length
        self.in_ch = args.sequence_length
        layers = []
        for ch in self.channels:
            layers.append(nn.Conv2d(in_ch, ch, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(p=self.dropout_rate))
            in_ch = ch
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.channels[-1], num_classes)
        self.dropout_fc = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):

        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
    
        x = self.feature_extractor(x)
        x = x.mean(dim=[2,3])
        x = self.dropout_fc(x)
        return self.classifier(x)

# -------------------------
# 2. LeNet5
# -------------------------
class LeNet5(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        d = getattr(args, "dropout", 0.5)
        self.in_ch = args.sequence_length
        self.conv1 = nn.Conv2d(self.in_ch, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(d)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.dropout(self.relu(self.fc1(x)))
        #x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.fc2(x))
        return self.fc3(x)

# -------------------------
# 3. AlexNet 
# -------------------------
class AlexNetClassifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        d = getattr(args, "dropout", 0.5)
        self.in_ch = args.sequence_length
        self.features = models.alexnet(pretrained=False).features
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1
        )

        self.dropout = nn.Dropout(d)
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 64),
            nn.ReLU(True),
            #nn.Dropout(d),
            #nn.Linear(4096, 4096),
            #nn.ReLU(True),
            nn.Dropout(d),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.features(x)
        #print(x.shape)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)
        ###print('after_flatten',x.shape)
        return self.classifier(x)

class AlexNetClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        d = getattr(args, "dropout", 0.5)
        self.in_ch = args.sequence_length
        # Load pretrained AlexNet
        self.features = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
        ).features
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.dropout = nn.Dropout(d)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 64),
            nn.ReLU(True),
            #nn.Dropout(d),
            #nn.Linear(4096, 4096),
            #nn.ReLU(True),
            nn.Dropout(d),
            nn.Linear(64, num_classes)
        )
        for param in self.features[0].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True            
    def forward(self, x):
        # Convert grayscale → RGB
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)x = torch.flatten(x, 1)
        return self.classifier(x)

class AlexNetClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        d = getattr(args, "dropout", 0.5)
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # User chooses depth
        
        # 1. Load and Slice
        full_alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        limit = min(self.requested_layers, len(full_alexnet))
        self.features = nn.Sequential(*list(full_alexnet.children())[:limit])

        # 2. Replace first conv layer
        self.features[0] = nn.Conv2d(self.in_ch, 64, kernel_size=5, stride=2, padding=1)

        # 3. SET ALL TO TRAINABLE (No freezing loop needed)
        for param in self.features.parameters():
            param.requires_grad = True

        # 4. Dynamic Head Setup
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        with torch.no_grad():
            dummy_out = self.features(torch.zeros(1, self.in_ch, 224, 224))
            in_channels_at_depth = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_channels_at_depth * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(d),
            nn.Linear(256, num_classes)
        )
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        if len(x.shape) == 5: x = x.squeeze(1)  
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1: x = x.repeat(1, self.in_ch, 1, 1)
        
        x = self.features(x)
        x = self.avgpool(x)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)x = torch.flatten(x, 1)x = torch.flatten(x, 1)
        return self.classifier(x)

class AlexNetClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        d = getattr(args, "dropout", 0.5)
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Up to 13
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Load pretrained AlexNet features
        full_alexnet = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
        ).features
        
        # 2. Slice the features based on num_layers
        # AlexNet features has 13 elements total
        limit = min(self.requested_layers, len(full_alexnet))
        self.features = nn.Sequential(*list(full_alexnet.children())[:limit])

        # 3. Replace first conv layer for 5-channel sequence input
        # Standard AlexNet features[0] is a 11x11 conv; keeping your 5x5 preference
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1
        )

        # 4. FREEZE all layers initially
        for param in self.features.parameters():
            param.requires_grad = False

        # 5. DYNAMIC UNFREEZING
        # Always unfreeze the modified input layer
        for param in self.features[0].parameters():
            param.requires_grad = True

        # Unfreeze the last N layers of the specified depth
        if self.num_to_train > 0:
            start_train_idx = max(0, limit - self.num_to_train)
            for i in range(start_train_idx, limit):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Pooling and Dynamic Head
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        # Detect in_features dynamically based on the depth slice
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_out = self.features(dummy_input)
            in_channels_at_depth = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_channels_at_depth * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(d),
            nn.Linear(256, num_classes)
        )
        
        # Classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Standardize input to 5-channel sequence
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
        
        x = self.features(x)
        x = self.avgpool(x)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)x = torch.flatten(x, 1)x = torch.flatten(x, 1)x = torch.flatten(x, 1)
        return self.classifier(x)
    
# -------------------------
# 4-5. VGG16 / VGG19
# -------------------------
class VGG16Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        self.features = models.vgg16(pretrained=False).features
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=1
        )        
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,128),
            #nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096,4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.features(x)
        #print(x.shape)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)x = torch.flatten(x, 1)x = torch.flatten(x, 1)x = torch.flatten(x,1)

        return self.classifier(x)
class VGG16Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        # ---- Pretrained VGG16 ----
        self.features = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        ).features

        # ---- Grayscale input ----

        # ---- Freeze backbone ----
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=1
        ) 
        # ---- Adaptive pooling ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Trainable classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,128),
            #nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096,4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(128,num_classes)
        )

        for param in self.features[0].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.mean(dim=[2,3])#x = torch.flatten(x, 1)
        return self.classifier(x)
class VGG16Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Depth in the feature list
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Load pretrained VGG16 features
        full_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # 2. Slice features based on num_layers
        # VGG16 features list has 31 elements
        limit = min(self.requested_layers, len(full_vgg))
        self.features = nn.Sequential(*list(full_vgg.children())[:limit])

        # 3. Replace first conv layer for 5-channel sequence input
        # Standard VGG uses kernel_size=3; changed from your 5 to maintain VGG structure
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 4. FREEZE all layers initially
        for param in self.features.parameters():
            param.requires_grad = False

        # 5. DYNAMIC UNFREEZING
        # Always unfreeze the modified input layer
        for param in self.features[0].parameters():
            param.requires_grad = True

        # Unfreeze the last N layers of the specified depth
        if self.num_to_train > 0:
            start_train_idx = max(0, limit - self.num_to_train)
            for i in range(start_train_idx, limit):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Pooling and Dynamic Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine in_features dynamically based on where we sliced the model
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_out = self.features(dummy_input)
            in_features = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        
        # Classifier head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Standardize input to 5 channels
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            else:
                # Custom logic for 3-channel to 5-channel if needed
                pass

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class VGG16Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length  # 5
        self.requested_layers = args.num_layers  # User defines how many layers to keep
        
        # 1. Load pretrained VGG16 features
        full_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # 2. Slice the features
        # VGG16 features list has 31 elements. We take only what is requested.
        limit = min(self.requested_layers, len(full_vgg))
        self.features = nn.Sequential(*list(full_vgg.children())[:limit])

        # 3. Replace the first conv layer (Mandatory for 5-channel input)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 4. TRAIN ALL LAYERS
        # Since we do not loop through parameters to set requires_grad=False,
        # every layer in self.features and the replaced conv1 are trainable by default.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Adaptive Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 6. Dynamic Head Calculation
        # We must detect the output channels of the last layer in our slice
        with torch.no_grad():
            dummy_out = self.features(torch.zeros(1, self.in_ch, 224, 224))
            in_features = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        
        # Classifier is also trainable
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Ensure 5 channels
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG19Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        self.features = models.vgg19(pretrained=False).features      
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=1
        ) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,128),
            #nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096,4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)
    
class VGG19Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        # ---- Load pretrained VGG19 ----
        self.features = vgg19(
            weights=VGG19_Weights.IMAGENET1K_V1
        ).features

        # ---- Replace first conv layer for grayscale ----

        # ---- Freeze pretrained backbone ----
        for param in self.features.parameters():
            param.requires_grad = False
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=1
        ) 
        # ---- Adaptive pooling (fixes size issues) ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Trainable classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,128),
            #nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096,4096),
            #nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(128,num_classes)
        )

        for param in self.features[0].parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)  
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class VGG19Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Number of blocks/layers to include
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Load pretrained VGG19 features
        full_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        # 2. Slice the features based on num_layers
        # In VGG19, there are 37 entries in the features list.
        # We ensure the slice doesn't exceed the model length.
        limit = min(self.requested_layers, len(full_vgg))
        self.features = nn.Sequential(*list(full_vgg.children())[:limit])

        # 3. Replace/Update first conv layer for 5-channel input
        # Note: In VGG, features[0] is the first Conv2d
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=3, # Standard VGG uses 3x3
            stride=1,
            padding=1
        )

        # 4. FREEZE Logic
        for param in self.features.parameters():
            param.requires_grad = False

        # 5. DYNAMIC UNFREEZING
        # Always unfreeze the modified first layer
        for param in self.features[0].parameters():
            param.requires_grad = True

        # Unfreeze the last N layers of the slice
        if self.num_to_train > 0:
            start_train_idx = max(0, limit - self.num_to_train)
            for i in range(start_train_idx, limit):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Adaptive pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine the input features for the Linear layer dynamically
        # VGG layers typically end with 512 channels
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_out = self.features(dummy_input)
            in_features = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        
        # Classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match input to 5-channel sequence
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            else:
                # Handle standard 3-channel to 5-channel if necessary
                pass

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
class VGG19Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Number of layers to keep and train
        
        # 1. Load pretrained VGG19 features
        # VGG19 contains 37 modules in the features list
        full_vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # 2. Slice the features
        limit = min(self.requested_layers, len(full_vgg))
        self.features = nn.Sequential(*list(full_vgg.children())[:limit])

        # 3. Replace/Update first conv layer for 5-channel input
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # We skip the "param.requires_grad = False" loop entirely.
        # This makes every layer in our slice trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 6. Dynamic Linear Head
        # VGG19 channels vary (64, 128, 256, 512) depending on the slice limit
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_out = self.features(dummy_input)
            in_features = dummy_out.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        
        # Head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True 

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Ensure input matches model's 5-channel input
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# -------------------------
# 6-9. ResNet Variants
# -------------------------
class ResNet18Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        self.model = models.resnet18(pretrained=False)  
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        ##print(self.model)       
        self.model.fc = nn.Linear(self.model.fc.in_features,num_classes)
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1) 
        return self.model(x)
class ResNet18Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # -------- Modify first conv for 12 channels --------

        # Initialize conv1 from pretrained weights
        # -------- FREEZE ALL CONVOLUTION / BACKBONE --------
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # -------- Replace and UNFREEZE classifier --------
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.conv1.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)      
        return self.model(x)
class ResNet18Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Total blocks to keep (e.g., 3)
        num_to_train = args.num_lastlayerstrain  # Blocks at the end to unfreeze (e.g., 2)
        # 1. Load pretrained ResNet18
        full_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        all_children = list(full_resnet.children())
        
        # 2. Slice the model
        # Stem is index 0-3. Blocks are index 4-7.
        stem = all_children[:4]
        blocks = all_children[4 : 4 + requested_layers] 
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING FIRST
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. UNFREEZE the first Conv layer (Modified for your sequence)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE: Last 'n' blocks
        # We look only at the blocks we kept (self.features[4:])
        # If requested_layers=3 and num_to_train=2, we unfreeze blocks at index 5 and 6
        if num_to_train > 0:
            # We skip the stem (4 layers) and unfreeze from the end of the blocks list
            for i in range(len(self.features) - num_to_train, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        out_channels_map = {1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map[requested_layers]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Head is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"Model Summary: {requested_layers} total blocks.")
        print(f"Trainable: First Conv + {num_to_train} last blocks + FC layer.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
class ResNet18Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        self.requested_layers = args.num_layers  # Number of stages (1 to 4)
        
        # 1. Load pretrained ResNet18
        full_resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        all_children = list(full_resnet.children())
        
        # 2. Slice the model
        # Stem (Conv, BN, ReLU, MaxPool) is index 0-3. 
        # Stages are index 4-7.
        stem = all_children[:4]
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_children[4 : 4 + limit] 
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace the first Conv layer
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # We skip the "requires_grad = False" loop. 
        # Every parameter in the stem and chosen stages will be trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # ResNet18 channels: Stage 1=64, Stage 2=128, Stage 3=256, Stage 4=512
        out_channels_map = {1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map[limit]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Head is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"Model Summary: ResNet18 sliced to {limit} stages. All layers are trainable.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Standardize input to sequence length
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    

class ResNet34Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        # ---- NO pretrained weights ----
        self.model = models.resnet34(weights=None)
        ##print(self.model)
        # ---- Modify first convolution for 1-channel input ----
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Replace classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)            
        return self.model(x)
class ResNet34Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        # ---- Load pretrained ResNet34 ----
        self.model = models.resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        # ---- Replace first conv for grayscale ----

        # ---- Freeze backbone ----
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Trainable classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.conv1.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ###print(x.shape)
        return self.model(x)
#ResNet34Classifier_Dynamic_Frozen
class ResNet34Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained ResNet34
        full_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 (Conv, BN, ReLU, MaxPool)
        # Blocks: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        blocks = all_children[4 : 4 + requested_layers] 
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer (The input bridge)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' blocks)
        # We look at the list of items in self.features.
        # If requested_layers is 3, self.features has 7 items (indices 0-6).
        # num_to_train=2 means we unfreeze index 6 and index 5.
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # ResNet34 output channels: L1=64, L2=128, L3=256, L4=512
        out_channels_map = {1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map[requested_layers]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # FC is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet34: Keeping {requested_layers} blocks. Unfreezing last {num_to_train} blocks + FC.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and x.shape[1] == 1:
            # Match the input channels we set in __init__
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
class ResNet34Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Number of stages (1 to 4)
        
        # 1. Load pretrained ResNet34
        full_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem (Conv, BN, ReLU, MaxPool) = index 0-3
        # Stages (layer1, layer2, layer3, layer4) = index 4-7
        stem = all_children[:4]
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_children[4 : 4 + limit] 
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace/Modify First Layer for 5-channel input
        # Note: We do this AFTER creating Sequential so we can access features[0]
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS (No Freezing)
        # We ensure everything is trainable. PyTorch sets True by default, 
        # but we can be explicit here to match your logic.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # ResNet34 Stage Output Channels: L1=64, L2=128, L3=256, L4=512
        out_channels_map = {1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map[limit]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure FC is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet34 Sliced: Using {limit} stages. All layers are TRAINING.")

    def forward(self, x):
        # Shape: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Handle single channel repetition if necessary
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    

class ResNet50Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        self.in_ch = args.sequence_length
        # ---- NO pretrained weights ----
        self.model = models.resnet50(weights=None)
        ##print(self.model)
        # ---- Modify first convolution for 1-channel input ----
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Replace classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)            
        return self.model(x)
class ResNet50Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length
        # ---- Load pretrained ResNet34 ----
        self.model = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1
        )

        # ---- Replace first conv for grayscale ----

        # ---- Freeze backbone ----
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Trainable classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.conv1.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ###print(x.shape)
        return self.model(x)   
class ResNet50Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained ResNet50
        full_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 | Blocks: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        blocks = all_children[4 : 4 + requested_layers] 
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING FIRST
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' blocks based on your logic)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head (ResNet50 has much wider channels)
        # Layer 1 -> 256 | Layer 2 -> 512 | Layer 3 -> 1024 | Layer 4 -> 2048
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map[requested_layers]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # FC is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet50: {requested_layers} blocks active. Training first layer + last {num_to_train} blocks.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and x.shape[1] == 1:
            # Repeat grayscale to match expected sequence channels
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
class ResNet50Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 4
        
        # 1. Load pretrained ResNet50
        full_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 (Conv1, BN, ReLU, MaxPool)
        # Stages: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_children[4 : 4 + limit] 
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Modify First Layer for 5-channel sequence input
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected bottleneck stages is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Note the different channel mapping for ResNet50 (Bottleneck design)
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map[limit]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure FC is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet50 Sliced: Using {limit} stages. ALL layers active for training.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Handle single channel repetition
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class ResNet101Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- NO pretrained weights ----
        self.in_ch = args.sequence_length
        # ---- NO pretrained weights ----
        self.model = models.resnet101(weights=None)
        ##print(self.model)
        # ---- Modify first convolution for 1-channel input ----
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Replace classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)            
        return self.model(x)
class ResNet101Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        self.in_ch = args.sequence_length
        # ---- Load pretrained ResNet34 ----
        self.model = models.resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V1
        )

        # ---- Replace first conv for grayscale ----

        # ---- Freeze backbone ----
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # ---- Trainable classifier ----
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.conv1.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ###print(x.shape)
        return self.model(x)
class ResNet101Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained ResNet101
        full_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 | Blocks: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        blocks = all_children[4 : 4 + requested_layers] 
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING FIRST
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' blocks)
        # Counting backwards from the end of the self.features Sequential list
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # ResNet101 Bottleneck channels: L1=256, L2=512, L3=1024, L4=2048
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map[requested_layers]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # FC is always trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet101 Summary: {requested_layers} stages kept.")
        print(f"Trainable: Conv1, the last {num_to_train} stage(s), and the FC head.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and x.shape[1] == 1:
            # Repeat to match sequence input channels
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)    
class ResNet101Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 4
        
        # 1. Load pretrained ResNet101
        full_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 (Conv1, BN, ReLU, MaxPool)
        # Stages: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_children[4 : 4 + limit] 
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace First Layer for 5-channel sequence input
        # Standard ResNet uses 7x7 stride 2; using 3x3 stride 1 for better patch detail
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected stages is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # ResNet101 uses Bottleneck layers: L1=256, L2=512, L3=1024, L4=2048
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map[limit]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure FC is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet101 Sliced: Keeping {limit} stages. ALL layers are TRAINABLE.")

    def forward(self, x):
        # Shape handling [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match expected input channels
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
# -------------------------
# 10-12. DenseNet Variants
# -------------------------
class DenseNet121Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )        
        self.model.classifier = nn.Linear(self.model.classifier.in_features,num_classes)
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet121Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained DenseNet121 ----
        self.model = models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )   
        # ---- Replace classifier (trainable) ----
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_classes
        )

        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features.conv0.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet121Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained DenseNet121
        full_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # DenseNet features are in model.features. 
        # Indices: 0-3 (Stem), 4 (DB1), 5 (Trans1), 6 (DB2), 7 (Trans2), 8 (DB3), 9 (Trans3), 10 (DB4), 11 (Final Norm)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stem is index 0-3 (conv0, norm0, relu0, pool0)
        stem = all_features[:4]
        
        # Mapping requested_layers to feature indices (Each layer is a DenseBlock + its Transition)
        # 1 layer -> DB1 (index 4)
        # 2 layers -> DB1, Trans1, DB2 (indices 4, 5, 6)
        # 3 layers -> DB1, Trans1, DB2, Trans2, DB3 (indices 4, 5, 6, 7, 8)
        # 4 layers -> DB1...DB4, Norm (indices 4 to 11)
        block_end_map = {1: 5, 2: 7, 3: 9, 4: 12}
        blocks = all_features[4 : block_end_map[requested_layers]]
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer (conv0)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' blocks)
        # In DenseNet, we count the DenseBlocks from the end of our sliced features
        if num_to_train > 0:
            # We unfreeze the last 'n' items in our Sequential list
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # DenseNet channels: DB1=256, DB2=512, DB3=1024, DB4=1024
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 1024}
        in_features = out_channels_map[requested_layers]
        
        self.classifier = nn.Linear(in_features, num_classes)
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"DenseNet121: {requested_layers} layers active. Training last {num_to_train} and Conv0.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)
class ResNet101Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 4
        
        # 1. Load pretrained ResNet101
        full_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        all_children = list(full_model.children())
        
        # 2. Slice the model
        # Stem: index 0-3 (Conv1, BN, ReLU, MaxPool)
        # Stages: index 4-7 (layer1, layer2, layer3, layer4)
        stem = all_children[:4]
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_children[4 : 4 + limit] 
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace First Layer for 5-channel sequence input
        # Standard ResNet uses 7x7 stride 2; using 3x3 stride 1 for better patch detail
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected stages is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # ResNet101 uses Bottleneck layers: L1=256, L2=512, L3=1024, L4=2048
        out_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map[limit]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure FC is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ResNet101 Sliced: Keeping {limit} stages. ALL layers are TRAINABLE.")

    def forward(self, x):
        # Shape handling [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match expected input channels
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    

class DenseNet169Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.densenet169(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )        
        self.model.classifier = nn.Linear(self.model.classifier.in_features,num_classes)
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet169Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained DenseNet121 ----
        self.model = models.densenet169(
            weights=DenseNet169_Weights.IMAGENET1K_V1
        )

        for param in self.model.parameters():
            param.requires_grad = False
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )   
        # ---- Replace classifier (trainable) ----
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_classes
        )

        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features.conv0.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet169Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained DenseNet169
        full_model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stem: indices 0-3 (conv0, norm0, relu0, pool0)
        stem = all_features[:4]
        
        # Mapping requested blocks to indices
        # Layer 1: DenseBlock 1 (index 4)
        # Layer 2: + Transition 1, DenseBlock 2 (indices 5, 6)
        # Layer 3: + Transition 2, DenseBlock 3 (indices 7, 8)
        # Layer 4: + Transition 3, DenseBlock 4, Norm5 (indices 9, 10, 11)
        block_end_map = {1: 5, 2: 7, 3: 9, 4: 12}
        blocks = all_features[4 : block_end_map[requested_layers]]
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer (conv0)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' items from the slice)
        # If requested_layers=3 and num_to_train=2, this unfreezes index 7 and 8
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # DenseNet169 Final channels: DB1=256, DB2=512, DB3=1280, DB4=1664
        # Note: DenseNet169 is wider than 121 at Layer 3 and 4
        out_channels_map = {1: 256, 2: 512, 3: 1280, 4: 1664}
        in_features = out_channels_map[requested_layers]
        
        self.classifier = nn.Linear(in_features, num_classes)
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"DenseNet169: Active Blocks={requested_layers}, Trainable Blocks={num_to_train}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)
class DenseNet169Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 4
        
        # 1. Load pretrained DenseNet169 features
        full_model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stem: indices 0-3 (conv0, norm0, relu0, pool0)
        stem = all_features[:4]
        
        # Mapping requested stages to block indices in DenseNet169
        # Stage 1: DenseBlock 1
        # Stage 2: + Transition 1, DenseBlock 2
        # Stage 3: + Transition 2, DenseBlock 3
        # Stage 4: + Transition 3, DenseBlock 4, Norm5
        block_end_map = {1: 5, 2: 7, 3: 9, 4: 12}
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_features[4 : block_end_map[limit]]
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace/Modify First Layer for 5-channel input
        # Using kernel 3, stride 1 to keep more spatial info for medical patches
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected DenseBlocks is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Mapping output channels based on the depth of the DenseNet
        out_channels_map = {1: 256, 2: 512, 3: 1280, 4: 1664}
        in_features = out_channels_map[limit]
        
        self.classifier = nn.Linear(in_features, num_classes)
        
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"DenseNet169 Sliced: {limit} stages kept. ALL layers are TRAINING.")

    def forward(self, x):
        # Shape handling: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match expected input channels
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        features = self.features(x)
        
        # DenseNet specific: Global Avg Pool requires final ReLU
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)
    

class DenseNet201Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.densenet201(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )        
        self.model.classifier = nn.Linear(self.model.classifier.in_features,num_classes)
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet201Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained DenseNet121 ----
        self.model = models.densenet201(
            weights=DenseNet201_Weights.IMAGENET1K_V1
        )

        for param in self.model.parameters():
            param.requires_grad = False
        self.in_ch = args.sequence_length
        self.model.features.conv0 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )   
        # ---- Replace classifier (trainable) ----
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_classes
        )

        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features.conv0.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class DenseNet201Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 3
        num_to_train = args.num_lastlayerstrain  # e.g., 2
        
        # 1. Load pretrained DenseNet201
        full_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stem: indices 0-3
        stem = all_features[:4]
        
        # Mapping requested blocks to indices
        # Layer 1: DB1 | Layer 2: +Trans1, DB2 | Layer 3: +Trans2, DB3 | Layer 4: +Trans3, DB4, Norm5
        block_end_map = {1: 5, 2: 7, 3: 9, 4: 12}
        blocks = all_features[4 : block_end_map[requested_layers]]
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer (conv0)
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        for param in self.features[0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' items)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # DenseNet201 channels: DB1=256, DB2=512, DB3=1792, DB4=1920
        out_channels_map = {1: 256, 2: 512, 3: 1792, 4: 1920}
        in_features = out_channels_map[requested_layers]
        
        self.classifier = nn.Linear(in_features, num_classes)
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"DenseNet201: Depth Stage {requested_layers}, Training Last {num_to_train} Items")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0].in_channels, 1, 1)
            
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)
class DenseNet201Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 4
        
        # 1. Load pretrained DenseNet201 features
        full_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stem: indices 0-3 (conv0, norm0, relu0, pool0)
        stem = all_features[:4]
        
        # Mapping requested stages to block indices in DenseNet201
        # Stage 1: DenseBlock 1
        # Stage 2: + Transition 1, DenseBlock 2
        # Stage 3: + Transition 2, DenseBlock 3
        # Stage 4: + Transition 3, DenseBlock 4, Norm5
        block_end_map = {1: 5, 2: 7, 3: 9, 4: 12}
        limit = max(1, min(self.requested_layers, 4))
        blocks = all_features[4 : block_end_map[limit]]
        
        self.features = nn.Sequential(*stem, *blocks)
        
        # 3. Replace/Modify First Layer for 5-channel input
        # We use a 3x3 kernel to preserve finer spatial details than the original 7x7
        self.features[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected DenseBlocks is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # DenseNet201 channels: DB1=256, DB2=512, DB3=1792, DB4=1920
        out_channels_map = {1: 256, 2: 512, 3: 1792, 4: 1920}
        in_features = out_channels_map[limit]
        
        self.classifier = nn.Linear(in_features, num_classes)
        
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"DenseNet201 Sliced: {limit} stages kept. ALL layers are TRAINING.")

    def forward(self, x):
        # Shape handling: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match expected input channels if input is single-channel
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        features = self.features(x)
        
        # DenseNet specific: Final feature maps must pass through ReLU before Pooling
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

# -------------------------
# 13-18. EfficientNet / MobileNet
# -------------------------
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,num_classes)
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB0Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)    
class EfficientNetB0Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Stages to keep (e.g., 1 to 9)
        num_to_train = args.num_lastlayerstrain  # Last 'n' stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B0
        full_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet features are a Sequential list of 9 stages
        # stage 0: Stem
        # stage 1-7: MBConv Blocks
        # stage 8: Final Conv + BN + Activation
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Always take the stem (index 0) and then the requested amount of blocks
        # requested_layers usually maps to stages 1 through 8.
        # If args.num_layers is 3, we take stage 0, 1, 2, 3.
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Layer (features[0][0])
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # Output channels for B0 stages: 1:16, 2:24, 3:40, 4:80, 5:112, 6:192, 7:320, 8:1280
        out_channels_map = {1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320, 8: 1280, 9: 1280}
        in_features = out_channels_map.get(requested_layers, 1280)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B0: {requested_layers} stages active. Unfreezing last {num_to_train}.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB0Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length  # 5
        self.requested_layers = args.num_layers  # Stages 1 to 8 (or 9)
        
        # 1. Load pretrained EfficientNet-B0
        full_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet features are a Sequential list of 9 stages (0 to 8)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0 is the Stem. Stages 1-7 are MBConv blocks. Index 8 is Final Conv.
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # Original: features[0][0] is a 3x3 Conv with 3 in_channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every stage from the new stem to the chosen MBConv block is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Output channels for B0 stages: 1:16, 2:24, 3:40, 4:80, 5:112, 6:192, 7:320, 8:1280
        out_channels_map = {1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320, 8: 1280}
        in_features = out_channels_map.get(limit, 1280)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B0 Sliced: {limit} stages kept. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match expected input channels (5) if input is single-channel
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
# ------------------ EfficientNet B1 ------------------
class EfficientNetB1Classifier(nn.Module):
    def __init__(self,args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )        
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB1Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b1(
            weights=EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        #print(self.model)
        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)  
class EfficientNetB1Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Number of stages to keep (max 8)
        num_to_train = args.num_lastlayerstrain  # Number of last stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B1
        full_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0 is the Stem. Indices 1-8 are the MBConv stages and final conv.
        # If num_layers=8, we take all_features[:9]
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # Note: B1 still starts with 32 out_channels in the stem
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # Output channels for B1 stages: 1:16, 2:24, 3:40, 4:80, 5:112, 6:192, 7:320, 8:1280
        out_channels_map = {1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320, 8: 1280}
        in_features = out_channels_map.get(requested_layers, 1280)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B1: Active Stages: {requested_layers}, Trainable: Last {num_to_train}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)    
class EfficientNetB1Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length  # 5
        self.requested_layers = args.num_layers  # Stages to keep (typically 1 to 8)
        
        # 1. Load pretrained EfficientNet-B1
        full_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0 is the Stem. Indices 1-7 are MBConv stages. Index 8 is the final conv stage.
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # B1 uses 32 out_channels in the first conv layer.
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # By not setting requires_grad = False, we ensure all sliced layers train.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Output channels for B1 stages: 1:16, 2:24, 3:40, 4:80, 5:112, 6:192, 7:320, 8:1280
        out_channels_map = {1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320, 8: 1280}
        in_features = out_channels_map.get(limit, 1280)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
        # Head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B1 Sliced: {limit} stages active. ALL layers training.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Standardize input for sequence channel length
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    


# ------------------ EfficientNet B2 ------------------
class EfficientNetB2Classifier(nn.Module):
    def __init__(self,args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )           
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB2Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b2(
            weights=EfficientNet_B2_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB2Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Max 8
        num_to_train = args.num_lastlayerstrain  # Last 'n' stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B2
        full_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Blocks | Index 8: Final Conv
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B2 Output channels: 1:16, 2:24, 3:48, 4:88, 5:120, 6:208, 7:352, 8:1408
        out_channels_map = {1: 16, 2: 24, 3: 48, 4: 88, 5: 120, 6: 208, 7: 352, 8: 1408}
        in_features = out_channels_map.get(requested_layers, 1408)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3), # B2 usually uses slightly higher dropout
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B2: Sliced to stage {requested_layers}. Last {num_to_train} stages trainable.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB2Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages to keep (typically 1 to 8)
        
        # 1. Load pretrained EfficientNet-B2
        full_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0 is the Stem. Stages 1-7 are MBConv stages. Index 8 is the final conv.
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # B2 uses 32 out_channels in the first conv layer (stage 0, index 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # We ensure every parameter in our slice is active for the optimizer
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Output channels for B2 stages: 1:16, 2:24, 3:48, 4:88, 5:120, 6:208, 7:352, 8:1408
        out_channels_map = {1: 16, 2: 24, 3: 48, 4: 88, 5: 120, 6: 208, 7: 352, 8: 1408}
        in_features = out_channels_map.get(limit, 1408)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3), # B2 standard dropout
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B2 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match expected input channels (5) if input is single-channel
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    

# ------------------ EfficientNet B3 ------------------
class EfficientNetB3Classifier(nn.Module):
    def __init__(self, args,num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=40, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )   
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB3Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=40, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB3Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Max 8
        num_to_train = args.num_lastlayerstrain  # Number of stages from end to unfreeze
        
        # 1. Load pretrained EfficientNet-B3
        full_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Stages 0-8. Index 0 is the stem.
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # B3 Stem still uses 40 out_channels (unlike B0-B2 which use 32)
        # We must match the original architecture's output channel count for the stem
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=40, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B3 Output channels: 1:24, 2:32, 3:48, 4:96, 5:136, 6:232, 7:384, 8:1536
        out_channels_map = {1: 24, 2: 32, 3: 48, 4: 96, 5: 136, 6: 232, 7: 384, 8: 1536}
        in_features = out_channels_map.get(requested_layers, 1536)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3), # Recommended dropout for B3
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B3: Sliced to stage {requested_layers}. Training last {num_to_train} stages.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB3Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 8
        
        # 1. Load pretrained EfficientNet-B3
        full_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Stages | Index 8: Final Conv
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # CRITICAL: B3 uses 40 out_channels in the stem (Stage 0, Item 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=40, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and chosen stages is active for training.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # B3 Channels: 1:24, 2:32, 3:48, 4:96, 5:136, 6:232, 7:384, 8:1536
        out_channels_map = {1: 24, 2: 32, 3: 48, 4: 96, 5: 136, 6: 232, 7: 384, 8: 1536}
        in_features = out_channels_map.get(limit, 1536)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
        # Head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B3 Sliced: Stage {limit} reached. ALL layers training.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ------------------ EfficientNet B4 ------------------
class EfficientNetB4Classifier(nn.Module):
    def __init__(self,args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )   
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB4Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB4Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Stages to keep (0-8)
        num_to_train = args.num_lastlayerstrain  # Last 'n' stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B4
        full_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Stage 0 is the Stem. Stages 1-8 are MBConv and final conv.
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # CRITICAL: B4 Stem expects 48 output channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B4 Output channels: 1:24, 2:32, 3:56, 4:112, 5:160, 6:272, 7:448, 8:1792
        out_channels_map = {1: 24, 2: 32, 3: 56, 4: 112, 5: 160, 6: 272, 7: 448, 8: 1792}
        in_features = out_channels_map.get(requested_layers, 1792)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), # B4 uses higher dropout (0.4) for regularization
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B4: Layers {requested_layers}, Unfrozen {num_to_train}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB4Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 8
        
        # 1. Load pretrained EfficientNet-B4
        full_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Stages | Index 8: Final Conv
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # CRITICAL: B4 Stem uses 48 out_channels (Stage 0, Item 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected stages is trainable.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # B4 Channels: 1:24, 2:32, 3:56, 4:112, 5:160, 6:272, 7:448, 8:1792
        out_channels_map = {1: 24, 2: 32, 3: 56, 4: 112, 5: 160, 6: 272, 7: 448, 8: 1792}
        in_features = out_channels_map.get(limit, 1792)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), # B4 standard dropout is higher to prevent overfitting
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B4 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
# ------------------ EfficientNet B5 ------------------
class EfficientNetB5Classifier(nn.Module):
    def __init__(self,args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )   
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB5Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b5(
            weights=EfficientNet_B5_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB5Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Max 8
        num_to_train = args.num_lastlayerstrain  # Number of stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B5
        full_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Stage 0: Stem | Stages 1-7: MBConv | Stage 8: Final Conv
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # B5 Stem expects 48 out_channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B5 Output channels: 1:24, 2:40, 3:64, 4:128, 5:176, 6:304, 7:512, 8:2048
        out_channels_map = {1: 24, 2: 40, 3: 64, 4: 128, 5: 176, 6: 304, 7: 512, 8: 2048}
        in_features = out_channels_map.get(requested_layers, 2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), # B5 uses high dropout
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B5: Stages {requested_layers} Active, Last {num_to_train} Trainable.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB5Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 8
        
        # 1. Load pretrained EfficientNet-B5
        full_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Stages | Index 8: Final Conv
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # B5 Stem expects 48 out_channels (similar to B4)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=48, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # B5 Channels: 1:24, 2:40, 3:64, 4:128, 5:176, 6:304, 7:512, 8:2048
        out_channels_map = {1: 24, 2: 40, 3: 64, 4: 128, 5: 176, 6: 304, 7: 512, 8: 2048}
        in_features = out_channels_map.get(limit, 2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), 
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B5 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x) 
# ------------------ EfficientNet B6 ------------------
class EfficientNetB6Classifier(nn.Module):
    def __init__(self, args,num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=56, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )   
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB6Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b6(
            weights=EfficientNet_B6_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=56, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB6Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Stages to keep (0-8)
        num_to_train = args.num_lastlayerstrain  # Last 'n' stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B6
        full_model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers (Stages 0-8)
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # B6 Stem requires exactly 56 out_channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=56, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B6 Output channels: 1:32, 2:40, 3:72, 4:144, 5:200, 6:344, 7:576, 8:2304
        out_channels_map = {1: 32, 2: 40, 3: 72, 4: 144, 5: 200, 6: 344, 7: 576, 8: 2304}
        in_features = out_channels_map.get(requested_layers, 2304)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # B6 uses high dropout (0.5) for heavy regularization
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B6: Sliced to stage {requested_layers}. Last {num_to_train} stages trainable.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB6Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 8
        
        # 1. Load pretrained EfficientNet-B6
        full_model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Stages | Index 8: Final Conv expansion
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # CRITICAL: B6 Stem expects 56 out_channels (Stage 0, Item 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=56, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and chosen stages is active for training.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # B6 Channels: 1:32, 2:40, 3:72, 4:144, 5:200, 6:344, 7:576, 8:2304
        out_channels_map = {1: 32, 2: 40, 3: 72, 4: 144, 5: 200, 6: 344, 7: 576, 8: 2304}
        in_features = out_channels_map.get(limit, 2304)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # B6 standard dropout is high to handle its high capacity
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B6 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5) if single-channel input is provided
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)    
# ------------------ EfficientNet B7 ------------------
class EfficientNetB7Classifier(nn.Module):
    def __init__(self,args, num_classes=1000):
        super().__init__()
        self.model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )   
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB7Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()

        # ---- Load pretrained EfficientNet-B0 ----
        self.model = models.efficientnet_b7(
            weights=EfficientNet_B7_Weights.IMAGENET1K_V1
        )

        # ---- Freeze entire backbone ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Replace classifier (trainable) ----
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        # ---- Explicitly unfreeze classifier ----
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class EfficientNetB7Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Stages to keep (0-8)
        num_to_train = args.num_lastlayerstrain  # Last 'n' stages to unfreeze
        
        # 1. Load pretrained EfficientNet-B7
        full_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers (Stages 0-8)
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stage 0, Item 0)
        # B7 Stem requires exactly 64 out_channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(start_unfreeze, len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # B7 Output channels stage-by-stage: 
        # 1:32, 2:48, 3:80, 4:160, 5:224, 6:384, 7:640, 8:2560
        out_channels_map = {1: 32, 2: 48, 3: 80, 4: 160, 5: 224, 6: 384, 7: 640, 8: 2560}
        in_features = out_channels_map.get(requested_layers, 2560)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # B7 uses high dropout (0.5)
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B7: Stage {requested_layers} active. Training {num_to_train} stages + Conv0.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.features[0][0].in_channels, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
class EfficientNetB7Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Stages 1 to 8
        
        # 1. Load pretrained EfficientNet-B7
        full_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice based on num_layers
        # Index 0: Stem | Indices 1-7: MBConv Stages | Index 8: Final Conv expansion
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Replace/Modify the first Conv layer in the Stem
        # CRITICAL: B7 Stem expects exactly 64 out_channels (Stage 0, Item 0)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the stem and the selected stages is active for training.
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # B7 Channels: 1:32, 2:48, 3:80, 4:160, 5:224, 6:384, 7:640, 8:2560
        out_channels_map = {1: 32, 2: 48, 3: 80, 4: 160, 5: 224, 6: 384, 7: 640, 8: 2560}
        in_features = out_channels_map.get(limit, 2560)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # B7 requires heavy regularization
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"EfficientNet-B7 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5)
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
 
# ------------------ MobileNetV2 ------------------
class MobileNetV2Classifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):
        # Convert grayscale to 3-channel
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class MobileNetV2Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False
        # Replace classifier (trainable)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )        
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)
class MobileNetV2Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # e.g., 2
        num_to_train = args.num_lastlayerstrain  # e.g., 1
        
        # 1. Load pretrained MobileNetV2
        full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Index 0 is stem, 1-17 are blocks, 18 is final conv
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stem)
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(max(0, start_unfreeze), len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. AUTO-DETECT Classifier Input Shape
        # We run a tiny dummy tensor through the features to see what comes out
        self.eval() # Set to eval mode for shape check
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            # Flattened size after Global Average Pooling is just the channel count
            in_features = dummy_output.shape[1] 
        self.train() # Set back to train mode

        # 7. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout if hasattr(args, 'dropout') else 0.2),
            nn.Linear(in_features, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV2: Sliced at {requested_layers}. Detected in_features: {in_features}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Ensure input channels match our new stem (e.g. 5 channels)
        if len(x.shape) == 4 and x.shape[1] != self.features[0][0].in_channels:
            # If input is 1-channel, repeat it. If it's already 5, this skip.
            if x.shape[1] == 1:
                x = x.repeat(1, self.features[0][0].in_channels, 1, 1)

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)
class MobileNetV2Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers 
        
        # 1. Load pretrained MobileNetV2
        full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Index 0 is stem, 1-17 are inverted residual blocks, 18 is final conv
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Modify Stem for 5-channel input
        # MobileNetV2 stem always outputs 32 channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the lightweight blocks will be updated
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. AUTO-DETECT Classifier Input Shape
        self.eval() 
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.shape[1] 
        self.train() 

        # 6. Classifier Head
        dropout_p = args.dropout if hasattr(args, 'dropout') else 0.2
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV2 Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching for 5-channel stem
        if len(x.shape) == 4 and x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

#    
# ------------------ MobileNetV3 Small ------------------
class MobileNetV3SmallClassifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)

class MobileNetV3SmallClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier (trainable)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )        
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)

class MobileNetV3SmallClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Slicing index
        num_to_train = args.num_lastlayerstrain  # Layers to unfreeze
        
        # 1. Load pretrained MobileNetV3 Small
        full_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # MobileNetV3 Small features has 13 items (0-12)
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING initially
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stem)
        # MobileNetV3 Small stem expects 16 out_channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(max(0, start_unfreeze), len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. AUTO-DETECT Classifier Input Shape
        # This prevents the "mat1 and mat2 shapes cannot be multiplied" error
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            # MobileNetV3 usually has an adaptive pool here; we check channels
            in_features = dummy_output.shape[1] 
        self.train()

        # 7. Classifier Head
        # MobileNetV3 uses a more sophisticated head than V2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=args.dropout if hasattr(args, 'dropout') else 0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV3-Small: Sliced at {requested_layers}. Detected in_features: {in_features}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching logic
        if len(x.shape) == 4 and x.shape[1] != self.features[0][0].in_channels:
            if x.shape[1] == 1:
                x = x.repeat(1, self.features[0][0].in_channels, 1, 1)

        x = self.features(x)
        # Pooling to reduce spatial dimensions to 1x1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

class MobileNetV3SmallClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers 
        
        # 1. Load pretrained MobileNetV3 Small
        full_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # V3-Small features has 13 items (0-12)
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Modify Stem for 5-channel input
        # MobileNetV3 Small stem (features[0]) always outputs 16 channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the lightweight blocks and SE modules will update
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. AUTO-DETECT Classifier Input Shape
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.shape[1] 
        self.train()

        # 6. Classifier Head
        # Keeping the V3-specific architecture: Linear -> Hardswish -> Dropout -> Linear
        dropout_p = args.dropout if hasattr(args, 'dropout') else 0.2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(1024, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV3-Small Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching for 5-channel stem
        if len(x.shape) == 4 and x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
# ------------------ MobileNetV3 Large ------------------
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=False)
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)

class MobileNetV3LargeClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier (trainable)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features,
            num_classes
        )
        self.in_ch = args.sequence_length
        self.model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )        
        for param in self.model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        return self.model(x)

class MobileNetV3LargeClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length
        requested_layers = args.num_layers       # Max 15
        num_to_train = args.num_lastlayerstrain  # Number of layers to unfreeze
        
        # 1. Load pretrained MobileNetV3 Large
        full_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Index 0: Stem | Indices 1-14: Bneck blocks | Index 15: Final 1x1 conv
        self.features = nn.Sequential(*all_features[:requested_layers + 1])
        
        # 3. FREEZE EVERYTHING initially
        for param in self.features.parameters():
            param.requires_grad = False
            
        # 4. MODIFY & UNFREEZE First Conv (Stem)
        # CRITICAL: V3-Large Stem expects 16 output channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        for param in self.features[0][0].parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' items)
        if num_to_train > 0:
            start_unfreeze = len(self.features) - num_to_train
            for i in range(max(0, start_unfreeze), len(self.features)):
                for param in self.features[i].parameters():
                    param.requires_grad = True

        # 6. AUTO-DETECT Classifier Input Shape
        # This fixes the "mat1 and mat2 shapes cannot be multiplied" error
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.shape[1] 
        self.train()

        # 7. Classifier Head
        # V3-Large uses a wider expansion (1280) in the head compared to Small
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=args.dropout if hasattr(args, 'dropout') else 0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV3-Large: Sliced at {requested_layers}. Detected in_features: {in_features}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Match channels for multi-sequence or grayscale input
        if len(x.shape) == 4 and x.shape[1] != self.features[0][0].in_channels:
            if x.shape[1] == 1:
                x = x.repeat(1, self.features[0][0].in_channels, 1, 1)

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

class MobileNetV3LargeClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers 
        
        # 1. Load pretrained MobileNetV3 Large
        full_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        all_features = list(full_model.features.children())
        
        # 2. Slice the model
        # Index 0: Stem | Indices 1-14: Bneck blocks | Index 15: Final expansion
        limit = max(1, min(self.requested_layers, len(all_features) - 1))
        self.features = nn.Sequential(*all_features[:limit + 1])
        
        # 3. Modify Stem for 5-channel input
        # V3-Large stem (features[0]) outputs 16 channels
        self.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Allows full fine-tuning of the Inverted Residual blocks and SE attention
        for param in self.features.parameters():
            param.requires_grad = True

        # 5. AUTO-DETECT Classifier Input Shape
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.shape[1] 
        self.train()

        # 6. Classifier Head
        # V3-Large uses a wider internal expansion (1280) than the Small version
        dropout_p = args.dropout if hasattr(args, 'dropout') else 0.2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(1280, num_classes)
        )
        
        # Ensure head is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        print(f"MobileNetV3-Large Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching logic
        if len(x.shape) == 4 and x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)    
# -------------------------
# 19-20. Transformers
# -------------------------

class ViTClassifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        config = ViTConfig(num_channels=self.in_ch, num_hidden_layers=3)        
        self.model = ViTModel(config)
        
        # Update projection for 5 channels
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=768,
            kernel_size=(16, 16), 
            stride=(16, 16)
        )

        # 1. First, set everything to NOT trainable
        #for param in self.model.parameters():
         #   param.requires_grad = False

        # 2. Unfreeze the Projection (Input)
        #for param in self.model.embeddings.patch_embeddings.projection.parameters():
        #    param.requires_grad = True

        # 3. Unfreeze the last 2 Transformer layers (so it can actually learn patterns)
        # Standard ViT has 'layer' list in 'encoder'
        #for i in range(2, 3): # Unfreezes layers 3 and 4 (the last two)
        #    for param in self.model.encoder.layer[i].parameters():
        #        param.requires_grad = True

        # 4. Unfreeze the Head (Output)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)
        #for param in self.fc.parameters():
        #    param.requires_grad = True

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching logic
        if len(x.shape) == 4 and x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        outputs = self.model(x).last_hidden_state
        cls_token = outputs[:, 0]
        return self.fc(cls_token)

class ViTClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        config = ViTConfig(num_channels=self.in_ch, num_hidden_layers=3)        
        self.model = ViTModel(config)
        
        # Update projection for 5 channels
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=768,
            kernel_size=(16, 16), 
            stride=(16, 16)
        )

        # 1. First, set everything to NOT trainable
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze the Projection (Input)
        for param in self.model.embeddings.patch_embeddings.projection.parameters():
            param.requires_grad = True

        # 3. Unfreeze the last 2 Transformer layers (so it can actually learn patterns)
        # Standard ViT has 'layer' list in 'encoder'
        for i in range(2, 3): # Unfreezes layers 3 and 4 (the last two)
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True

        # 4. Unfreeze the Head (Output)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Ensure x has 5 channels before interpolate
        # If your data is [B, 1, H, W], repeat it to 5, not 3.
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        outputs = self.model(x).last_hidden_state
        cls_token = outputs[:, 0]
        return self.fc(cls_token)

class ViTClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Initialize Config with the CORRECT channel count
        # This prevents the ValueError by informing the internal validation
        config = ViTConfig(
            num_channels=self.in_ch,            # SET TO 5
            num_hidden_layers=args.num_layers, 
            image_size=224,
            patch_size=16
        )
        
        # 2. Initialize model with config
        self.model = ViTModel(config)
        
        # 3. Manually re-verify Projection (Hugging Face sometimes resets this)
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=config.hidden_size, # 768
            kernel_size=(16, 16), 
            stride=(16, 16)
        )

        # 4. FREEZE EVERYTHING
        for param in self.model.parameters():
            param.requires_grad = False

        # 5. UNFREEZE Projection (Input)
        for param in self.model.embeddings.patch_embeddings.projection.parameters():
            param.requires_grad = True

        # 6. DYNAMIC UNFREEZE (Transformer blocks)
        num_to_train = args.num_lastlayerstrain
        total_layers = len(self.model.encoder.layer)
        if num_to_train > 0:
            start_index = max(0, total_layers - num_to_train)
            for i in range(start_index, total_layers):
                for param in self.model.encoder.layer[i].parameters():
                    param.requires_grad = True

        # 7. Classifier Head
        self.fc = nn.Linear(config.hidden_size, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input to the expected 5 channels
        expected_ch = self.model.embeddings.patch_embeddings.projection.in_channels
        if x.shape[1] != expected_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, expected_ch, 1, 1)

        # ViT requires 224x224 to match the 16x16 patch projection
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Forward pass through Transformer
        outputs = self.model(x).last_hidden_state
        cls_token = outputs[:, 0] # Take the first token
        return self.fc(cls_token)

class ViTClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Initialize Config with the 5-channel count
        # args.num_layers defines how many Transformer blocks to initialize
        config = ViTConfig(
            num_channels=self.in_ch,
            num_hidden_layers=args.num_layers, 
            image_size=224,
            patch_size=16,
            hidden_size=768,      # Standard ViT-Base hidden size
            num_attention_heads=12
        )
        
        # 2. Initialize model with config
        self.model = ViTModel(config)
        
        # 3. Ensure the Patch Projection matches 5-channel input
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=config.hidden_size,
            kernel_size=(16, 16), 
            stride=(16, 16)
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # This includes embeddings, position encodings, and all encoder blocks
        for param in self.model.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        self.fc = nn.Linear(config.hidden_size, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"ViT Encoder: {args.num_layers} Transformer blocks active. ALL layers TRAINING.")

    def forward(self, x):
        # Handle [Batch, 1, 5, H, W] -> [Batch, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input to the expected 5 channels (repeat grayscale if needed)
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        # ViT requires fixed spatial resolution for patch projection
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Pass through Transformer encoder
        outputs = self.model(x).last_hidden_state
        
        # Extract the [CLS] token (index 0) which aggregates the sequence info
        cls_token = outputs[:, 0] 
        return self.fc(cls_token)

class SwinClassifier(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        
        # 1. Define Config: 1 stage with 2 layers = 2 layers total
        # This prevents unnecessary downsampling (Patch Merging)
        config = SwinConfig(
            num_channels=self.in_ch,
            depths=[2],                # One stage with 2 layers
            embed_dim=96,              # Initial embedding dimension
            num_heads=[3],             # Heads for the first stage
            window_size=7              # Standard window size
        )
        
        self.model = SwinModel(config)
        
        # 2. Update projection for 5 channels
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=96,
            kernel_size=(4, 4), 
            stride=(4, 4)
        )
        
        # 3. Classifier head matches the embed_dim of the last stage
        self.fc = nn.Linear(self.model.num_features, num_classes)

    def forward(self, x):
        # [Batch, 1, 5, H, W] -> [Batch, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
            
        # Standardize size for Swin
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Forward pass
        outputs = self.model(x).last_hidden_state
        
        # Global Average Pooling (Pooling across the sequence of patches)
        out = outputs.mean(dim=1) 
        
        return self.fc(out)

class SwinClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=1000):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        
        # 1. Define Config: 2 stages with 1 layer each = 2 layers total
        # We also set num_channels=5 to avoid the ValueError
        config = SwinConfig(
            num_channels=self.in_ch,
            depths=[1, 1, 0, 0],       # Only 2 layers total
            embed_dim=96,              # Standard Tiny Swin dimension
            num_heads=[3, 6, 12, 24]   # Standard heads
        )
        
        # 2. Initialize model with the specific config
        self.model = SwinModel(config)
        
        # 3. Update the projection layer to handle 5 channels
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=96,
            kernel_size=(4, 4), 
            stride=(4, 4)
        )
        
        # Swin hidden size for the last stage (Stage 1 in this 2-layer case) is 192
        # Because Stage 0 (96) -> Patch Merging -> Stage 1 (192)
        # self.model.num_features automatically finds the correct final dimension
        #self.fc = nn.Linear(self.model.num_features, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False        
        self.fc = nn.Linear(self.model.num_features, num_classes)
        for param in self.fc.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True
    def forward(self, x):
        # Handle [Batch, 1, Seq, H, W] -> [Batch, Seq, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
            
        # Resize to 224x224 (Swin default)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Swin outputs [Batch, Sequence_Length, Channels]
        # We perform Global Average Pooling over the sequence (dim 1)
        outputs = self.model(x).last_hidden_state
        out = outputs.mean(dim=1) 
        
        return self.fc(out)
    
class SwinClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # e.g., 5
        # 1. Define Hierarchical Config
        # Swin depth is defined per stage. 
        # For a small custom model, we distribute num_layers across stages.
        # depth [2, 2, 6, 2] is standard Tiny. Here we use a simpler logic.
        config = SwinConfig(
            num_channels=self.in_ch,
            depths=[args.num_layers] * 2, # Example: 2 layers in stage 1, 2 in stage 2
            embed_dim=96,
            image_size=224,
            patch_size=4
        )
        # 2. Initialize with Config
        self.model = SwinModel(config)
        # 3. Update Projection Layer
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=config.embed_dim, # 96
            kernel_size=(4, 4), 
            stride=(4, 4)
        )
        # 4. FREEZE EVERYTHING
        for param in self.model.parameters():
            param.requires_grad = False
        # 5. UNFREEZE Projection (Input)
        for param in self.model.embeddings.patch_embeddings.projection.parameters():
            param.requires_grad = True
        # 6. DYNAMIC UNFREEZE (Last 'n' Stages/Layers)
        num_to_train = args.num_lastlayerstrain
        # Swin layers are inside self.model.encoder.layers
        total_stages = len(self.model.encoder.layers)
        if num_to_train > 0:
            start_stage = max(0, total_stages - num_to_train)
            for i in range(start_stage, total_stages):
                for param in self.model.encoder.layers[i].parameters():
                    param.requires_grad = True
        # 7. Classifier Head
        # Swin's final dimension is embed_dim * 2^(num_stages - 1)
        self.fc = nn.Linear(self.model.num_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True
        print(f"Swin: {total_stages} stages active. Unfreezing last {num_to_train} stages.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        # Ensure x matches the 5 channels (in_ch)
        expected_ch = self.model.embeddings.patch_embeddings.projection.in_channels
        if x.shape[1] != expected_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, expected_ch, 1, 1)
        # Swin requires 224x224
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Swin Forward
        # last_hidden_state: [Batch, L, Hidden_Dim] (where L is total patches)
        outputs = self.model(x).last_hidden_state
        # Global Average Pooling over the patch sequence
        out = outputs.mean(dim=1) 
        
        return self.fc(out)

class SwinClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Define Hierarchical Config
        # Standard depths for Swin-Tiny: [2, 2, 6, 2]. 
        # Here we use your dynamic num_layers across 4 stages.
        config = SwinConfig(
            num_channels=self.in_ch,
            depths=[args.num_layers] * 4, 
            embed_dim=96,
            image_size=224,
            patch_size=4,
            num_heads=[3, 6, 12, 24]
        )
        
        # 2. Initialize with Config
        self.model = SwinModel(config)
        
        # 3. Update Projection Layer for 5-channel input
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=config.embed_dim, # 96
            kernel_size=(4, 4), 
            stride=(4, 4)
        )

        # 4. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Includes Patch Embeddings, Stage Merging layers, and Window Attention blocks
        for param in self.model.parameters():
            param.requires_grad = True

        # 5. Classifier Head
        # Swin's final dimension (num_features) is embed_dim * 2^(num_stages - 1)
        # For embed_dim 96 and 4 stages: 96 * 8 = 768
        self.fc = nn.Linear(self.model.num_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"Swin Transformer: {len(config.depths)} stages active. ALL layers TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5)
        expected_ch = self.model.embeddings.patch_embeddings.projection.in_channels
        if x.shape[1] != expected_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, expected_ch, 1, 1)

        # Swin requires 224x224 to align with window sizes
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Swin Forward pass
        # last_hidden_state: [Batch, Sequence_Length, Hidden_Dim]
        outputs = self.model(x).last_hidden_state
        
        # Global Average Pooling over the patch sequence (L dimension)
        out = outputs.mean(dim=1) 
        
        return self.fc(out)
# 21-27. Segmentation-based classifiers
# -------------------------
class UNetClassifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        # UNet with ResNet34 encoder
        self.in_ch = args.sequence_length # This is 5
        self.unet = Unet(
            encoder_name='resnet34',
            encoder_weights=None,  # None → train from scratch
            in_channels=self.in_ch,
            classes=1
        )
        # Linear classifier
        #print(self.unet)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Repeat grayscale → 3 channels
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        ##print(x.shape)
        # Extract encoder features
        features = self.unet.encoder(x)[-1]  # last layer
        pooled = features.mean(dim=[2, 3])   # global average pooling 
        return self.fc(pooled)
    
class UNetClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        # UNet with ResNet34 encoder
        self.in_ch = args.sequence_length # This is 5
        self.unet = Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1
        )

        # Freeze encoder parameters
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

        # Linear classifier (trainable)
        self.fc = nn.Linear(512, num_classes)
        for param in self.fc.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True
            
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)

        features = self.unet.encoder(x)[-1]
        pooled = features.mean(dim=[2, 3])
        return self.fc(pooled)
    
class UNetClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        requested_layers = args.num_layers       # Range: 0 to 4
        num_to_train = args.num_lastlayerstrain  # How many of the kept stages to unfreeze

        # 1. Initialize the Full UNet Encoder
        full_unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1
        )
        
        # 2. Slice the Encoder Stages
        # ResNet34 Encoder has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # We group them into logical "stages"
        self.stem = nn.Sequential(
            full_unet.encoder.conv1,
            full_unet.encoder.bn1,
            full_unet.encoder.relu,
            full_unet.encoder.maxpool
        )
        
        # Stages 1, 2, 3, 4
        self.stages = nn.ModuleList([
            full_unet.encoder.layer1,
            full_unet.encoder.layer2,
            full_unet.encoder.layer3,
            full_unet.encoder.layer4
        ])
        
        # Keep only up to requested_layers (0 means just the stem)
        self.active_stages = self.stages[:requested_layers]

        # 3. FREEZE EVERYTHING initially
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.active_stages.parameters():
            param.requires_grad = False

        # 4. UNFREEZE Stem (because we changed in_channels to 5)
        for param in self.stem.parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            # Unfreeze layers from the end of our sliced active_stages
            for i in range(len(self.active_stages) - num_to_train, len(self.active_stages)):
                if i >= 0:
                    for param in self.active_stages[i].parameters():
                        param.requires_grad = True

        # 6. Classifier Head
        # ResNet34 Channel Map: Stage 0=64, 1=64, 2=128, 3=256, 4=512
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(requested_layers, 512)
        
        self.fc = nn.Linear(in_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"UNet-Encoder: Sliced at stage {requested_layers}. Last {num_to_train} stages trainable.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Handle grayscale repeat to match in_ch (5)
        if x.shape[1] == 1:
            x = x.repeat(1, self.stem[0].in_channels, 1, 1)

        # Forward pass through sliced encoder
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)
class UNetClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Range: 0 to 4
        
        # 1. Initialize the UNet to extract the encoder
        # We use resnet34 as the backbone by default
        full_unet = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet', # Using pretrained weights
            in_channels=self.in_ch,
            classes=1
        )
        
        # 2. Extract and Slice the Encoder Stages
        # Stem: conv1, bn1, relu, maxpool
        self.stem = nn.Sequential(
            full_unet.encoder.conv1,
            full_unet.encoder.bn1,
            full_unet.encoder.relu,
            full_unet.encoder.maxpool
        )
        
        # Encoder stages: layer1, layer2, layer3, layer4
        all_stages = [
            full_unet.encoder.layer1,
            full_unet.encoder.layer2,
            full_unet.encoder.layer3,
            full_unet.encoder.layer4
        ]
        
        # Limit the number of stages based on num_layers
        limit = max(0, min(self.requested_layers, 4))
        self.active_stages = nn.ModuleList(all_stages[:limit])
        
        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # We ensure the stem and all selected encoder layers are trainable
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.active_stages.parameters():
            param.requires_grad = True

        # 4. Classifier Head
        # ResNet34 Channel Map: Stage 0=64, 1=64, 2=128, 3=256, 4=512
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(limit, 512)
        
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure head is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"UNet-Encoder Sliced: Stage {limit} active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Match input channels (5) if necessary
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # Pass through stem
        x = self.stem(x)
        
        # Pass through requested encoder stages
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling (convert spatial maps to vector)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class UNetPPClassifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        self.unetpp = UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=self.in_ch, classes=1)
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.unetpp.encoder(x)[-1]
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)
class UNetPPClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        self.unetpp = UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=self.in_ch, classes=1)
        for param in self.unetpp.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512,num_classes)
        for param in self.fc.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.unetpp.encoder(x)[-1]
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)
class UNetPPClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        requested_layers = args.num_layers       # Range: 0 to 4
        num_to_train = args.num_lastlayerstrain  # How many stages to unfreeze

        # 1. Initialize U-Net++ 
        # We use UnetPlusPlus but only interact with the encoder
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1
        )

        # 2. Slice the Encoder into stages
        # Stage 0: Initial Conv/Pool
        self.stem = nn.Sequential(
            self.model.encoder.conv1,
            self.model.encoder.bn1,
            self.model.encoder.relu,
            self.model.encoder.maxpool
        )
        
        # Stages 1-4: Residual Blocks
        self.stages = nn.ModuleList([
            self.model.encoder.layer1,
            self.model.encoder.layer2,
            self.model.encoder.layer3,
            self.model.encoder.layer4
        ])
        
        # Keep only up to requested_layers
        self.active_stages = self.stages[:requested_layers]

        # 3. FREEZE EVERYTHING initially
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # 4. UNFREEZE Stem (Crucial for 5-channel input)
        for param in self.stem.parameters():
            param.requires_grad = True

        # 5. DYNAMIC UNFREEZE (Last 'n' stages)
        if num_to_train > 0:
            # Calculate starting point relative to active_stages
            start_idx = len(self.active_stages) - num_to_train
            for i in range(max(0, start_idx), len(self.active_stages)):
                for param in self.active_stages[i].parameters():
                    param.requires_grad = True

        # 6. Classifier Head
        # ResNet34 Stage Channels: 0:64, 1:64, 2:128, 3:256, 4:512
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(requested_layers, 512)
        
        self.fc = nn.Linear(in_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"UNet++ (ResNet34): Stages active: {requested_layers}. Trainable stages: {num_to_train}.")

    def forward(self, x):
        # Handle Batch x Sequence x C x H x W
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching for 5-channel stem
        if x.shape[1] == 1:
            x = x.repeat(1, self.stem[0].in_channels, 1, 1)

        # Encoder pass
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)
class UNetPPClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Range: 0 to 4
        
        # 1. Initialize U-Net++ to extract the ResNet34 encoder
        # We use pretrained ImageNet weights as a starting point
        full_model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet', 
            in_channels=self.in_ch,
            classes=1
        )

        # 2. Slice the Encoder into stages
        # Stem: Initial Conv/BN/ReLU/Pool
        self.stem = nn.Sequential(
            full_model.encoder.conv1,
            full_model.encoder.bn1,
            full_model.encoder.relu,
            full_model.encoder.maxpool
        )
        
        # Stages 1-4: Residual Blocks
        all_stages = [
            full_model.encoder.layer1,
            full_model.encoder.layer2,
            full_model.encoder.layer3,
            full_model.encoder.layer4
        ]
        
        # Keep only up to requested_layers
        limit = max(0, min(self.requested_layers, 4))
        self.active_stages = nn.ModuleList(all_stages[:limit])

        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # All parameters in the stem and active residual blocks will train.
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.active_stages.parameters():
            param.requires_grad = True

        # 4. Classifier Head
        # ResNet34 Channel Map: 0:64, 1:64, 2:128, 3:256, 4:512
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(limit, 512)
        
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure head is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"UNet++ Encoder Sliced: {limit} stages active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        
        # Channel matching for 5-channel stem if input is grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # Forward pass through sliced encoder
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling (Spatial Reduction)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class AttentionUNetClassifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        # Use UnetPlusPlus with attention_type='scse' (supported in SMP)
        self.in_ch = args.sequence_length # This is 5
        self.attunet = UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1,
            attention_type='scse'
        )
        self.fc = nn.Linear(512,num_classes)

    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.attunet.encoder(x)[-1]
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)
class AttentionUNetClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        # Use UnetPlusPlus with attention_type='scse' (supported in SMP)
        self.in_ch = args.sequence_length
        self.attunet = UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1,
            attention_type='scse'
        )
        for param in self.attunet.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512,num_classes)
        for param in self.fc.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.attunet.encoder(x)[-1]
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)

class AttentionUNetClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        requested_layers = args.num_layers # 0-4
        num_to_train = args.num_lastlayerstrain
        
        # 1. Initialize Attention Unet++
        # scse = Concurrent Spatial and Channel Squeeze & Excitation
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=self.in_ch,
            classes=1,
            attention_type='scse' 
        )

        # 2. Slice the Encoder Stages
        self.stem = nn.Sequential(
            self.model.encoder.conv1,
            self.model.encoder.bn1,
            self.model.encoder.relu,
            self.model.encoder.maxpool
        )
        
        self.stages = nn.ModuleList([
            self.model.encoder.layer1,
            #self.model.encoder.layer2,
            #self.model.encoder.layer3,
            #self.model.encoder.layer4
        ])
        
        self.active_stages = self.stages[:requested_layers]

        # 3. Handle Training Logic (Freezing/Unfreezing)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze stem for 5-ch input
        for param in self.stem.parameters():
            param.requires_grad = True

        # Unfreeze last 'n' active stages
        if num_to_train > 0:
            start_idx = len(self.active_stages) - num_to_train
            for i in range(max(0, start_idx), len(self.active_stages)):
                for param in self.active_stages[i].parameters():
                    param.requires_grad = True

        # 4. Classifier Head
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(requested_layers, 512)
        
        self.fc = nn.Linear(in_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"Attention UNet++: Active Stages {requested_layers}, Trainable {num_to_train}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)            
        # 5. Execute Encoder
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # NOTE: If you want to use the 'scse' attention, it usually requires 
        # passing through the decoder blocks. However, for a classifier, 
        # we usually apply a custom Attention layer here if the encoder doesn't have it.
        
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class AttentionUNetClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        requested_layers = args.num_layers # 0-4
        
        # 1. Initialize Attention Unet++
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet', # Added imagenet weights as it's standard for Train-All
            in_channels=self.in_ch,
            classes=1,
            attention_type='scse' 
        )

        # 2. Slice the Encoder Stages
        self.stem = nn.Sequential(
            self.model.encoder.conv1,
            self.model.encoder.bn1,
            self.model.encoder.relu,
            self.model.encoder.maxpool
        )
        
        self.stages = nn.ModuleList([
            self.model.encoder.layer1,
            #self.model.encoder.layer2,
            #self.model.encoder.layer3,
            #self.model.encoder.layer4
        ])
        
        self.active_stages = self.stages[:requested_layers]

        # 3. NO FREEZING - ENABLE ALL GRADIENTS
        for param in self.parameters():
            param.requires_grad = True

        # 4. Classifier Head
        out_channels_map = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        in_features = out_channels_map.get(requested_layers, 512)
        
        self.fc = nn.Linear(in_features, num_classes)

        print(f"Attention UNet++: Active Stages {requested_layers}, TRAINING ALL LAYERS")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)             
        # 5. Execute Encoder
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
        
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class DeepLabV3Classifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length
        self.deeplab = deeplabv3_resnet50(pretrained=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        #print(self.deeplab)
        self.fc = nn.Linear(2048,num_classes)
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.deeplab.backbone(x)['out']
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)
class DeepLabV3Classifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length
        self.deeplab = deeplabv3_resnet50(pretrained=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        for param in self.deeplab.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(2048,num_classes)
        for param in self.deeplab.backbone.conv1.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True
        for param in self.fc.parameters():  # Ensure only classifier is trainable
            param.requires_grad = True            
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.deeplab.backbone(x)['out']
        pooled = features.mean(dim=[2,3])
        return self.fc(pooled)
class DeepLabV3Classifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        requested_layers = args.num_layers       # Range: 0 to 4
        num_to_train = args.num_lastlayerstrain  # Stages to unfreeze
        
        # 1. Initialize DeepLabV3 with ResNet50 backbone
        # We don't use pretrained=True because we changed the input channels to 5
        full_model = deeplabv3_resnet50(weights=None)
        backbone = full_model.backbone
        
        # 2. Modify the first conv layer for 5-channel input
        backbone.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )

        # 3. Slice the Backbone into Stages
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # ResNet50 Stages
        self.stages = nn.ModuleList([
            backbone.layer1, # Stage 1 (256 channels)
            backbone.layer2, # Stage 2 (512 channels)
            backbone.layer3, # Stage 3 (1024 channels)
            backbone.layer4  # Stage 4 (2048 channels)
        ])
        
        self.active_stages = self.stages[:requested_layers]

        # 4. FREEZE Logic
        for param in self.parameters():
            param.requires_grad = False
            
        # Always unfreeze the 5-channel stem
        for param in self.stem.parameters():
            param.requires_grad = True

        # Dynamic Unfreeze for last 'n' stages
        if num_to_train > 0:
            start_idx = len(self.active_stages) - num_to_train
            for i in range(max(0, start_idx), len(self.active_stages)):
                for param in self.active_stages[i].parameters():
                    param.requires_grad = True

        # 5. Classifier Head
        # ResNet50 channel map: Stage 0:64, 1:256, 2:512, 3:1024, 4:2048
        out_channels_map = {0: 64, 1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map.get(requested_layers, 2048)
        
        self.fc = nn.Linear(in_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"DeepLabV3-ResNet50: Stages active: {requested_layers}. Trainable: {num_to_train}")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels
        if x.shape[1] == 1:
            x = x.repeat(1, self.stem[0].in_channels, 1, 1)

        # Pass through sliced backbone
        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)   
class DeepLabV3Classifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Range: 0 to 4
        
        # 1. Initialize DeepLabV3 and extract the backbone
        # We use weights to get the pretrained ResNet50 features
        full_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        backbone = full_model.backbone
        
        # 2. Slice the Backbone into Stages
        # Stem: Conv1, BN1, ReLU, Maxpool
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # Modify Stem for 5-channel input
        # Note: We do this after extraction to ensure we replace the correct layer
        self.stem[0] = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        
        # ResNet50 Stages (layers 1-4)
        all_stages = [
            backbone.layer1, # Stage 1: 256 channels
            backbone.layer2, # Stage 2: 512 channels
            backbone.layer3, # Stage 3: 1024 channels (Dilated in DeepLab)
            backbone.layer4  # Stage 4: 2048 channels (Dilated in DeepLab)
        ]
        
        limit = max(0, min(self.requested_layers, 4))
        self.active_stages = nn.ModuleList(all_stages[:limit])

        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.active_stages.parameters():
            param.requires_grad = True

        # 4. Classifier Head
        # Mapping: 0:64, 1:256, 2:512, 3:1024, 4:2048
        out_channels_map = {0: 64, 1: 256, 2: 512, 3: 1024, 4: 2048}
        in_features = out_channels_map.get(limit, 2048)
        
        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure head is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"DeepLabV3 Backbone Sliced: Stage {limit} active. ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
            
        # Match input channels (5) if single-channel input is provided
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        x = self.stem(x)
        for stage in self.active_stages:
            x = stage(x)
            
        # Global Average Pooling
        x = x.mean(dim=[2, 3])
        return self.fc(x)
    

class HRNetClassifier(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False):
        super().__init__()
        # Load HRNet backbone
        self.in_ch = args.sequence_length
        self.model = timm.create_model('hrnet_w18', pretrained=pretrained, num_classes=0)  # no classifier
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        #print(self.model)
        self.fc = nn.Linear(self.model.num_features, num_classes)
        
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        features = self.model(x)
        return self.fc(features)
class HRNetClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10, pretrained=False):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Load HRNet backbone (w18 is the lightweight version)
        self.model = timm.create_model('hrnet_w18', pretrained=pretrained, num_classes=0)
        # 2. Update first layer to accept 5 channels
        self.model.conv1 = nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        # 3. FREEZE the backbone
        for param in self.model.parameters():
            param.requires_grad = False
        # 4. UNFREEZE specific layers so the model can actually learn
        # Unfreeze the new 5-channel input layer
        for param in self.model.conv1.parameters():
            param.requires_grad = True
        # 5. Define and Unfreeze the classifier head
        self.fc = nn.Linear(self.model.num_features, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Handle [Batch, 1, 5, H, W] -> [Batch, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        # FIX: Repeat grayscale to 5 channels (matches in_ch)
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        # HRNet keeps high-resolution representations throughout
        features = self.model(x)
        return self.fc(features)
class HRNetClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # 1 to 4
        num_to_train = args.num_lastlayerstrain
        
        # 1. Load HRNet backbone
        self.model = timm.create_model('hrnet_w18', pretrained=False, num_classes=0)
        
        # 2. Update first layer
        self.model.conv1 = nn.Conv2d(self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 3. Freeze all initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 4. Unfreeze logic
        # Always unfreeze stem (conv1, bn1, conv2, bn2)
        for m in [self.model.conv1, self.model.bn1, self.model.conv2, self.model.bn2]:
            for param in m.parameters():
                param.requires_grad = True

        # Map num_layers to internal HRNet attributes for unfreezing
        stage_names = ['layer1', 'stage2', 'stage3', 'stage4']
        active_names = stage_names[:self.requested_layers]
        
        if num_to_train > 0:
            to_unfreeze = active_names[-num_to_train:]
            for name in to_unfreeze:
                layer = getattr(self.model, name)
                for param in layer.parameters():
                    param.requires_grad = True

        # 5. Classifier Head
        # HRNet w18 feature counts: Stage 1=256, Stage 2/3/4=270
        out_dim = 256 if self.requested_layers == 1 else 270
        self.fc = nn.Linear(out_dim, num_classes)
        
        print(f"HRNet-w18: Using {self.requested_layers} stages. Unfrozen: {num_to_train} stages + stem.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        # FIX: Repeat grayscale to 5 channels (matches in_ch)
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)

        # Replicating internal HRNet forward but stopping at requested_layers
        # Stem
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.act2(x)
        
        # Stage 1
        x = self.model.layer1(x)
        if self.requested_layers == 1:
            return self.fc(x.mean(dim=[2, 3]))

        # Transition 1 (Creates the 2nd branch)
        x = self.model.transition1([x])
        
        # Stage 2
        x = self.model.stage2(x)
        if self.requested_layers == 2:
            return self.fc(x[0].mean(dim=[2, 3]))

        # Transition 2 (Creates the 3rd branch)
        x = self.model.transition2(x)
        
        # Stage 3
        x = self.model.stage3(x)
        if self.requested_layers == 3:
            return self.fc(x[0].mean(dim=[2, 3]))

        # Transition 3 (Creates the 4th branch)
        x = self.model.transition3(x)
        
        # Stage 4
        x = self.model.stage4(x)
        # For the final stage, we take the highest-res branch [0]
        return self.fc(x[0].mean(dim=[2, 3]))
class HRNetClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=2):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.sequence_length = args.sequence_length # 5
        self.requested_layers = args.num_layers # Range 1 to 4
        
        # 1. Load HRNet-W18 backbone
        # We use num_classes=0 to get the feature extractor only
        self.model = timm.create_model('hrnet_w18', pretrained=True, num_classes=0)
        
        # 2. Adjust Stem for 5-channel sequence input
        # We replace the first convolution to accept our sequence length
        self.model.conv1 = nn.Conv2d(
            self.sequence_length, 64, kernel_size=3, stride=2, padding=1, bias=False
        )

        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # In this version, the stem, all transitions, and all parallel stages train
        for param in self.model.parameters():
            param.requires_grad = True

        # 4. FC Input dimension calculation based on parallel branches
        # Branch widths for w18: Stage1=256 (bottleneck), Stage2=18+36, Stage3=18+36+72, Stage4=18+36+72+144
        if self.requested_layers == 1:
            in_features = 256 
        elif self.requested_layers == 2:
            in_features = 18 + 36 
        elif self.requested_layers == 3:
            in_features = 18 + 36 + 72 
        else:
            in_features = 270 # Total concatenated width of all 4 resolution branches

        self.fc = nn.Linear(in_features, num_classes)
        
        # Ensure FC is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"HRNet-W18: {self.requested_layers} stages active. ALL parallel branches TRAINING.")

    def _fuse_and_pool(self, x_list):
        """Global Average Pool all parallel resolution branches and concatenate."""
        pooled = [torch.mean(branch, dim=[2, 3]) for branch in x_list]
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        # Handle sequence dimension: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)
        if len(x.shape) == 4 and  x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)            
        # --- STEM ---
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.act2(x)
        
        # --- STAGE 1 (Single Resolution) ---
        x = self.model.layer1(x)
        if self.requested_layers == 1:
            return self.fc(torch.mean(x, dim=[2, 3]))

        # --- TRANSITION 1: Splits into 2 resolution branches ---
        x_list = []
        for trans in self.model.transition1:
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
        x = x_list 
        
        # --- STAGE 2 (2 Parallel Branches) ---
        x = self.model.stage2(x)
        if self.requested_layers == 2:
            return self.fc(self._fuse_and_pool(x))

        # --- TRANSITION 2: Splits into 3 resolution branches ---
        y = []
        for i, trans in enumerate(self.model.transition2):
            if trans is not None:
                inp = x[i] if i < len(x) else x[-1]
                y.append(trans(inp))
            else:
                y.append(x[i])
        x = y 
        
        # --- STAGE 3 (3 Parallel Branches) ---
        x = self.model.stage3(x)
        if self.requested_layers == 3:
            return self.fc(self._fuse_and_pool(x))

        # --- TRANSITION 3: Splits into 4 resolution branches ---
        y = []
        for i, trans in enumerate(self.model.transition3):
            if trans is not None:
                inp = x[i] if i < len(x) else x[-1]
                y.append(trans(inp))
            else:
                y.append(x[i])
        x = y 
        
        # --- STAGE 4 (4 Parallel Branches) ---
        x = self.model.stage4(x)
        return self.fc(self._fuse_and_pool(x))

class SegFormerClassifier(nn.Module):
    def __init__(self, args, num_classes, backbone="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Load the model (using standard from_pretrained)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # 2. Update the first layer for 5 channels
        # SegFormer-B0 uses a Mix Transformer (MiT) encoder
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=32, # 32 is standard for B0 stage 1
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3)
        )
        # 3. FREEZE logic
        # 4. Custom Classifier head (Trainable)
        # B0 encoder hidden state for the last layer is 256
        self.classifier_head = nn.Linear(256, num_classes)
        # Ensure the head and the new input projection are trainable
    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        # Match the 5-channel input expectation
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        # SegFormer typically expects 512x512 or 224x224
        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        outputs = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        # Last encoder hidden state: [Batch, 256, H/32, W/32]
        features = outputs.last_hidden_state
        # Global Average Pooling to flatten spatial dimensions
        pooled = features.mean(dim=(2, 3)) 
        return self.classifier_head(pooled)
class SegFormerClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes, backbone="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Load the model (using standard from_pretrained)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # 2. Update the first layer for 5 channels
        # SegFormer-B0 uses a Mix Transformer (MiT) encoder
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=32, # 32 is standard for B0 stage 1
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3)
        )
        # 3. FREEZE logic
        for param in self.model.parameters():
            param.requires_grad = False
        # 4. Custom Classifier head (Trainable)
        # B0 encoder hidden state for the last layer is 256
        self.classifier_head = nn.Linear(256, num_classes)
        # Ensure the head and the new input projection are trainable
        for param in self.classifier_head.parameters():
            param.requires_grad = True
        for param in self.model.segformer.encoder.patch_embeddings[0].proj.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        
        # Match the 5-channel input expectation
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)

        # SegFormer typically expects 512x512 or 224x224
        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

        outputs = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Last encoder hidden state: [Batch, 256, H/32, W/32]
        features = outputs.last_hidden_state

        # Global Average Pooling to flatten spatial dimensions
        pooled = features.mean(dim=(2, 3)) 

        return self.classifier_head(pooled)
class SegFormerClassifier_Dynamic_Frozen(nn.Module):

    def __init__(self, args, num_classes, backbone="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.in_ch = args.sequence_length 
        
        # We still use SegformerModel (the encoder) but point it to your ADE weights
        # This extracts the backbone from the segmentation model safely
        self.segformer = SegformerModel.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True 
        )
        
        # ... rest of the conv1 modification and freeze logic ...

        # 2. Update the first patch embedding layer for 5-channel input
        # We replace the 'proj' convolution while keeping the stride and kernel size
        old_proj = self.segformer.encoder.patch_embeddings[0].proj
        self.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=old_proj.out_channels, # Usually 32 for B0
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding
        )

        # 3. FREEZE backbone
        for param in self.segformer.parameters():
            param.requires_grad = False

        # 4. Custom Classification Head
        # mit-b0 last hidden state dimension is 256
        self.classifier_head = nn.Linear(256, num_classes)

        # 5. UNFREEZE parts necessary for the 5-channel mapping
        for param in self.segformer.encoder.patch_embeddings[0].proj.parameters():
            param.requires_grad = True
        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x: [Batch, 1, 5, H, W] -> [Batch, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        
        # Expansion if input is grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # SegFormer Encoder pass
        outputs = self.segformer(pixel_values=x)
        
        # features: [Batch, 256, H/32, W/32]
        features = outputs.last_hidden_state

        # Global Average Pooling (Reduce spatial H/W dimensions to 1x1)
        pooled = features.mean(dim=(2, 3)) 

        return self.classifier_head(pooled)
        # Shape: [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        
        # Handle single channel inputs
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # SegFormer Forward pass
        outputs = self.segformer(pixel_values=x)
        
        # last_hidden_state shape: [Batch, 256, H/32, W/32]
        features = outputs.last_hidden_state

        # Global Average Pooling
        pooled = features.mean(dim=(2, 3)) 

        return self.classifier_head(pooled)

class SegFormerClassifier_Sliced_TrainAll(nn.Module):
    def __init__(self, args, num_classes, backbone="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        # SegFormer has 4 stages. limit determines how many stages we keep.
        self.requested_stages = args.num_layers 
        
        # 1. Initialize Segformer Encoder
        self.segformer = SegformerModel.from_pretrained(
            backbone,
            ignore_mismatched_sizes=True,
            use_safetensors=True 
        )
        
        # 2. Update the first patch embedding layer for 5-channel input
        old_proj = self.segformer.encoder.patch_embeddings[0].proj
        self.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=old_proj.out_channels, 
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding
        )

        # 3. DYNAMIC SLICING (Stage Logic)
        # SegFormer-B0 has 4 blocks in self.segformer.encoder.block
        # We slice the stages and the corresponding patch embeddings
        limit = max(1, min(self.requested_stages, 4))
        self.segformer.encoder.block = nn.ModuleList(list(self.segformer.encoder.block)[:limit])
        self.segformer.encoder.patch_embeddings = nn.ModuleList(list(self.segformer.encoder.patch_embeddings)[:limit])

        # 4. ENABLE GRADIENTS FOR ALL
        for param in self.segformer.parameters():
            param.requires_grad = True

        # 5. AUTO-DETECT Classifier Input Shape
        # Each stage in SegFormer increases the channel dimension: 32 -> 64 -> 160 -> 256
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 224, 224)
            # We manually pass through the sliced encoder to find the output dimension
            outputs = self.segformer(pixel_values=dummy_input)
            in_features = outputs.last_hidden_state.shape[1]
        self.train()

        # 6. Classifier Head
        self.classifier_head = nn.Linear(in_features, num_classes)
        
        print(f"SegFormer Sliced: {limit} stages active. Output features: {in_features}. ALL layers TRAINING.")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)

        # Forward pass through the SLICED encoder
        outputs = self.segformer(pixel_values=x)
        features = outputs.last_hidden_state

        # Global Average Pooling
        pooled = features.mean(dim=(2, 3)) 

        return self.classifier_head(pooled)
# -------------------------
# 28-29. Hybrid models
# -------------------------

class UNetResNetClassifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Initialize U-Net (Attention Generator)
        # Note: in_channels must be 5 to match your sequence
        self.unet = Unet(encoder_name='resnet34', encoder_weights=None, in_channels=self.in_ch, classes=1)
        # 2. Initialize ResNet (Classifier)
        # We also modify its first layer to accept 5 channels
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # 3. FREEZE Logic
    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        # Ensure input matches the 5-channel requirement
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
        # Step 1: Generate Attention Mask [B, 1, H, W]
        mask = self.unet(x)
        # Step 2: Apply mask to the original input (Element-wise multiplication)
        # This "highlights" the features the U-Net thinks are important
        x_attended = x * torch.sigmoid(mask)
        
        # Step 3: Classify the "cleaned" image
        return self.resnet(x_attended)
class UNetResNetClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        
        # 1. Initialize U-Net (Attention Generator)
        # Note: in_channels must be 5 to match your sequence
        self.unet = Unet(encoder_name='resnet34', encoder_weights=None, in_channels=self.in_ch, classes=1)
        
        # 2. Initialize ResNet (Classifier)
        # We also modify its first layer to accept 5 channels
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # 3. FREEZE Logic
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False
            
        # 4. UNFREEZE the final classification head
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        for param in self.resnet.conv1.parameters():
            param.requires_grad = True            
        # OPTIONAL: Unfreeze the U-Net if you want it to learn how to mask
        # for param in self.unet.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Ensure input matches the 5-channel requirement
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
            
        # Step 1: Generate Attention Mask [B, 1, H, W]
        mask = self.unet(x)
        
        # Step 2: Apply mask to the original input (Element-wise multiplication)
        # This "highlights" the features the U-Net thinks are important
        x_attended = x * torch.sigmoid(mask)
        
        # Step 3: Classify the "cleaned" image
        return self.resnet(x_attended)
class UNetResNetClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # 1 to 4 stages
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Initialize U-Net (Attention Generator)
        self.unet = Unet(
            encoder_name='resnet34', 
            encoder_weights=None, 
            in_channels=self.in_ch, 
            classes=1
        )
        
        # 2. Initialize ResNet (Classifier)
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 3. FREEZE Logic
        for param in self.parameters():
            param.requires_grad = False
            
        # 4. UNFREEZE specific layers based on user settings
        # Always unfreeze the modified ResNet conv1 and the FC head
        for param in self.resnet.conv1.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # Unfreeze ResNet stages (layer1, layer2, layer3, layer4) backward
        resnet_stages = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        if self.num_to_train > 0:
            # We unfreeze the last N stages out of the requested depth
            start_train_idx = max(0, self.requested_layers - self.num_to_train)
            for i in range(start_train_idx, self.requested_layers):
                for param in resnet_stages[i].parameters():
                    param.requires_grad = True

        # Keep U-Net trainable so it can learn to highlight cancer cells
        for param in self.unet.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            else:
                raise ValueError(f"Input channels mismatch")
            
        # Step 1: Generate Attention Mask
        mask = self.unet(x)
        mask_prob = torch.sigmoid(mask)
        x_attended = x * mask_prob
        
        # Step 2: DYNAMIC RESNET FORWARD PASS
        # Stem
        feat = self.resnet.conv1(x_attended)
        feat = self.resnet.bn1(feat)
        feat = self.resnet.relu(feat)
        feat = self.resnet.maxpool(feat)
        
        # Stages (layer1 to layer4)
        resnet_stages = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        limit = max(1, min(self.requested_layers, 4))
        
        for i in range(limit):
            feat = resnet_stages[i](feat)
            
        # Step 3: Head Processing
        # Global Average Pool
        pooled = self.resnet.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        
        # Adjust FC input if we cut the model early (ResNet-18 dims: 64, 128, 256, 512)
        if flat.shape[1] != self.resnet.fc.in_features:
            # Create a temporary projection if depth is reduced
            dynamic_fc = nn.Linear(flat.shape[1], self.resnet.fc.out_features).to(flat.device)
            return dynamic_fc(flat)
            
        return self.resnet.fc(flat)

class UNetResNetClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # 1 to 4 stages
        
        # 1. Initialize U-Net (Attention Mask Generator)
        # We use resnet34 as the encoder for the UNet part
        self.unet = Unet(
            encoder_name='resnet34', 
            encoder_weights='imagenet', # Pretrained encoder helps attention converge
            in_channels=self.in_ch, 
            classes=1 # Output a single probability mask
        )
        
        # 2. Initialize ResNet-18 (The Classifier)
        self.resnet = models.resnet18(pretrained=True)
        
        # Adjust ResNet stem for 5-channel sequence
        self.resnet.conv1 = nn.Conv2d(
            self.in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Both the UNet (attention) and ResNet (classification) will update
        for param in self.parameters():
            param.requires_grad = True
            
        # 4. Final Classification Head
        # Note: We initialize this here so the in_features matches stage 4 (512)
        self.resnet.fc = nn.Linear(512, num_classes)

        print(f"UNet-ResNet Attentive: {self.requested_layers} stages. Dual-Network training active.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Ensure x matches the 5 channels
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            
        # --- STEP 1: ATTENTION GENERATION ---
        # UNet produces a mask of the same H, W as input
        mask = self.unet(x)
        mask_prob = torch.sigmoid(mask)
        
        # Apply mask: The model now 'looks' only at what the UNet thinks is important
        x_attended = x * mask_prob
        
        # --- STEP 2: CLASSIFICATION ---
        # Stem
        feat = self.resnet.conv1(x_attended)
        feat = self.resnet.bn1(feat)
        feat = self.resnet.relu(feat)
        feat = self.resnet.maxpool(feat)
        
        # Dynamic ResNet Layers
        stages = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        limit = max(1, min(self.requested_layers, 4))
        
        for i in range(limit):
            feat = stages[i](feat)
            
        # --- STEP 3: HEAD ---
        pooled = self.resnet.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        
        # Handle dynamic linear layer for early-exit slices
        if flat.shape[1] != self.resnet.fc.in_features:
            # ResNet-18 channel widths: [64, 128, 256, 512]
            dynamic_fc = nn.Linear(flat.shape[1], self.resnet.fc.out_features).to(flat.device)
            return dynamic_fc(flat)
            
        return self.resnet.fc(flat)


class AttentionUNetEfficientNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Unet++ with scSE Attention
        # We set in_channels=self.in_ch (5) to match your input
        self.unet = UnetPlusPlus(
            encoder_name='resnet34', 
            encoder_weights=None, 
            in_channels=self.in_ch, 
            classes=1, 
            attention_type='scse'
        )
        # 2. EfficientNet B0 Classifier
        self.eff = models.efficientnet_b0(pretrained=False)
        # Update EfficientNet's first layer for 5 channels
        self.eff.features[0][0] = nn.Conv2d(
            self.in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Update the final classifier head
        self.eff.classifier[1] = nn.Linear(self.eff.classifier[1].in_features, num_classes)
        # 3. FREEZE Logic

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match input to 5-channel expectation
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
            
        # Generate the scSE-refined mask
        mask = self.unet(x)
        
        # Apply the attention gate
        x_gated = x * torch.sigmoid(mask)
        
        # Classify via EfficientNet
        return self.eff(x_gated)
class AttentionUNetEfficientNet_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # This is 5
        # 1. Unet++ with scSE Attention
        # We set in_channels=self.in_ch (5) to match your input
        self.unet = UnetPlusPlus(
            encoder_name='resnet34', 
            encoder_weights=None, 
            in_channels=self.in_ch, 
            classes=1, 
            attention_type='scse'
        )
        # 2. EfficientNet B0 Classifier
        self.eff = models.efficientnet_b0(pretrained=False)
        # Update EfficientNet's first layer for 5 channels
        self.eff.features[0][0] = nn.Conv2d(
            self.in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Update the final classifier head
        self.eff.classifier[1] = nn.Linear(self.eff.classifier[1].in_features, num_classes)
        # 3. FREEZE Logic
        for param in self.parameters():
            param.requires_grad = False
            
        # 4. UNFREEZE the Classifier Head
        for param in self.eff.features[0][0].parameters():
            param.requires_grad = True
        for param in self.eff.classifier[1].parameters():
            param.requires_grad = True

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match input to 5-channel expectation
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1,self.in_ch, 1, 1)
            
        # Generate the scSE-refined mask
        mask = self.unet(x)
        
        # Apply the attention gate
        x_gated = x * torch.sigmoid(mask)
        
        # Classify via EfficientNet
        return self.eff(x_gated)
class AttentionUNetEfficientNet_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Depth of EfficientNet features
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Unet++ with scSE (Gate Generator)
        self.unet = UnetPlusPlus(
            encoder_name='resnet34', 
            encoder_weights=None, 
            in_channels=self.in_ch, 
            classes=1, 
            attention_type='scse'
        )
        
        # 2. EfficientNet B0 (Main Backbone)
        self.eff = models.efficientnet_b0(pretrained=False)
        
        # Update first conv layer for 5 channels
        self.eff.features[0][0] = nn.Conv2d(
            self.in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 3. FREEZE everything initially
        for param in self.parameters():
            param.requires_grad = False
            
        # 4. DYNAMIC UNFREEZING
        # Always unfreeze the new input layer and the final classifier head
        for param in self.eff.features[0][0].parameters():
            param.requires_grad = True
        
        # EfficientNet-B0 features has 9 indices (0 to 8)
        # We unfreeze the last N layers based on num_lastlayerstrain
        if self.num_to_train > 0:
            start_idx = max(0, 9 - self.num_to_train)
            for i in range(start_idx, 9):
                for param in self.eff.features[i].parameters():
                    param.requires_grad = True

        # Unfreeze the classifier head
        self.eff.classifier[1] = nn.Linear(self.eff.classifier[1].in_features, num_classes)
        for param in self.eff.classifier[1].parameters():
            param.requires_grad = True

        # Keep UNet++ parameters trainable so the mask can adapt
        for param in self.unet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            else:
                raise ValueError(f"Input channels mismatch")
            
        # 1. Generate Attention Mask
        mask = self.unet(x)
        mask_prob = torch.sigmoid(mask)
        x_gated = x * mask_prob
        
        # 2. Dynamic EfficientNet Forward Pass
        # Instead of calling self.eff(x_gated), we manually go through requested stages
        # EfficientNet features has 9 blocks.
        feat = x_gated
        limit = min(self.requested_layers, len(self.eff.features))
        
        for i in range(limit):
            feat = self.eff.features[i](feat)
            
        # 3. Head Processing
        # If we stopped early (limit < 9), we still need to Pool and Classify
        # EfficientNet uses Global Average Pooling before the head
        pooled = self.eff.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        
        # Note: If limit < 9, the number of features might not match the FC layer.
        # We check the dimension here to avoid a crash.
        if flat.shape[1] != self.eff.classifier[1].in_features:
            # Dynamic projection to match the classifier head if layers are reduced
            temp_fc = nn.Linear(flat.shape[1], self.eff.classifier[1].out_features).to(flat.device)
            return temp_fc(flat)
            
        return self.eff.classifier(flat)

class AttentionUNetEfficientNet_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers # Range: 1 to 9
        
        # 1. Initialize U-Net++ with scSE attention
        # The scSE attention helps the gate focus on specific channels AND spatial locations
        self.unet = UnetPlusPlus(
            encoder_name='resnet34', 
            encoder_weights='imagenet', # Pretrained weights help the attention gate converge
            in_channels=self.in_ch, 
            classes=1, 
            attention_type='scse'
        )
        
        # 2. Initialize EfficientNet B0
        self.eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Update first conv layer for 5-channel sequence input
        self.eff.features[0][0] = nn.Conv2d(
            self.in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Every parameter in the U-Net++ and EfficientNet-B0 is now trainable
        for param in self.parameters():
            param.requires_grad = True

        # Update classifier head to match num_classes
        # EfficientNet-B0 final output channels is 1280
        self.eff.classifier[1] = nn.Linear(1280, num_classes)

        print(f"U-Net++ (scSE) + EfficientNet-B0: ALL layers are TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
            
        # Match input channels (5)
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
            
        # --- STEP 1: GENERATE ATTENTION GATE ---
        # The U-Net++ produces a mask that "gates" the input
        mask = self.unet(x)
        mask_prob = torch.sigmoid(mask)
        x_gated = x * mask_prob
        
        # --- STEP 2: DYNAMIC FEATURE EXTRACTION ---
        feat = x_gated
        # Clamp limit between 1 and total available features (9 blocks)
        limit = max(1, min(self.requested_layers, len(self.eff.features)))
        
        for i in range(limit):
            feat = self.eff.features[i](feat)
            
        # --- STEP 3: HEAD PROCESSING ---
        pooled = self.eff.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        
        # Handle dynamic linear projection if backbone is sliced early
        if flat.shape[1] != self.eff.classifier[1].in_features:
            # We create a local linear layer to match the flattened feature size
            dynamic_fc = nn.Linear(flat.shape[1], self.eff.classifier[1].out_features).to(flat.device)
            return dynamic_fc(flat)
            
        return self.eff.classifier(flat)


class CellposeClassifier(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Load Cellpose backbone
        cp_model = CellposeModel(gpu=torch.cuda.is_available(), model_type='cyto')
        self.backbone = cp_model.net 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 2. DYNAMICALLY REPLACE THE FIRST CONV LAYER
# 1. (Earlier in your code) Load backbone and target_dtype...
        
        # 2. DYNAMICALLY REPLACE THE FIRST CONV LAYER
        self.first_layer_path = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                path_parts = name.rsplit('.', 1)
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels, 
                    kernel_size=module.kernel_size, 
                    stride=module.stride, 
                    padding=module.padding,
                    bias=(module.bias is not None)
                ).to(self.target_dtype)
                
                if len(path_parts) > 1:
                    parent = dict(self.backbone.named_modules())[path_parts[0]]
                    setattr(parent, path_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                
                self.first_layer_path = name
                break

        # 3. FREEZE Backbone (Everything becomes requires_grad=False)

        # 4. UNFREEZE the specific First Layer (Make it trainable)
        if self.first_layer_path:
            # Retrieve the newly created layer using the saved path
            all_modules = dict(self.backbone.named_modules())
            first_layer = all_modules[self.first_layer_path]
            

        # 5. Trainable Head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
        #print(x.shape)
        # 1. Match Grid Size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x.to(self.target_dtype)

        # 2. Forward pass
        # The Style-Unet/ViT-SAM net returns a tuple: (mask, styles)
        # mask is [B, 3, H, W] -> This caused your error
        # styles is [B, 256] -> This is what we want!
        mask, styles = self.backbone(x)
        
        # 3. Ensure styles is the 256-dim vector
        # If styles is somehow spatial, pool it. If it's already [B, 256], we are good.
        if len(styles.shape) == 4:
            styles = styles.mean(dim=[2, 3])
            
        # Convert to Float32 for the linear head
        styles = styles.to(torch.float32)
        
        return self.fc(styles)
class CellposeClassifier_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Load Cellpose backbone
        cp_model = CellposeModel(gpu=torch.cuda.is_available(), model_type='cyto')
        self.backbone = cp_model.net 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 2. DYNAMICALLY REPLACE THE FIRST CONV LAYER
# 1. (Earlier in your code) Load backbone and target_dtype...
        
        # 2. DYNAMICALLY REPLACE THE FIRST CONV LAYER
        self.first_layer_path = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                path_parts = name.rsplit('.', 1)
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels, 
                    kernel_size=module.kernel_size, 
                    stride=module.stride, 
                    padding=module.padding,
                    bias=(module.bias is not None)
                ).to(self.target_dtype)
                
                if len(path_parts) > 1:
                    parent = dict(self.backbone.named_modules())[path_parts[0]]
                    setattr(parent, path_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                
                self.first_layer_path = name
                break

        # 3. FREEZE Backbone (Everything becomes requires_grad=False)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 4. UNFREEZE the specific First Layer (Make it trainable)
        if self.first_layer_path:
            # Retrieve the newly created layer using the saved path
            all_modules = dict(self.backbone.named_modules())
            first_layer = all_modules[self.first_layer_path]
            
            # Set requires_grad to True only for this layer's weights/bias
            for param in first_layer.parameters():
                param.requires_grad = True
            #print(f"Target layer {self.first_layer_path} is now trainable.")

        # 5. Trainable Head
        self.fc = nn.Linear(256, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True
        
        if self.first_layer_path:
            layers_dict = dict(self.backbone.named_modules())
            for param in layers_dict[self.first_layer_path].parameters():
                param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if x.shape[1] != self.in_ch:
            if x.shape[1] == 1:
                x = x.repeat(1, self.in_ch, 1, 1)
        #print(x.shape)
        # 1. Match Grid Size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x.to(self.target_dtype)

        # 2. Forward pass
        # The Style-Unet/ViT-SAM net returns a tuple: (mask, styles)
        # mask is [B, 3, H, W] -> This caused your error
        # styles is [B, 256] -> This is what we want!
        mask, styles = self.backbone(x)
        
        # 3. Ensure styles is the 256-dim vector
        # If styles is somehow spatial, pool it. If it's already [B, 256], we are good.
        if len(styles.shape) == 4:
            styles = styles.mean(dim=[2, 3])
            
        # Convert to Float32 for the linear head
        styles = styles.to(torch.float32)
        
        return self.fc(styles)
class CellposeClassifier_Dynamic_Frozenn(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        self.requested_layers = args.num_layers 
        self.num_to_train = args.num_lastlayerstrain
        
        # 1. Load Cellpose backbone
        # model_type is handled internally by Cellpose now
        cp_model = CellposeModel(gpu=torch.cuda.is_available())
        self.backbone = cp_model.net 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 2. DYNAMIC INPUT LAYER REPLACEMENT
        # We find the first Conv2d regardless of whether it's UNet or Transformer
        self.first_layer_name = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                # Replace the first conv found
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None)
                ).to(self.target_dtype)
                
                # Use rsplit to find the parent module and set the new layer
                name_parts = name.rsplit('.', 1)
                if len(name_parts) > 1:
                    parent = dict(self.backbone.named_modules())[name_parts[0]]
                    setattr(parent, name_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                
                self.first_layer_name = name
                break

        # 3. FREEZE logic
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 4. UNFREEZE based on num_lastlayerstrain
        # Always unfreeze the modified input layer
        if self.first_layer_name:
            first_layer = dict(self.backbone.named_modules())[self.first_layer_name]
            for param in first_layer.parameters():
                param.requires_grad = True

        # Unfreeze the last N blocks of the backbone
        if self.num_to_train > 0:
            # We target 'blocks' for Transformer or 'down/up' for UNet
            trainable_candidates = []
            for name, module in self.backbone.named_modules():
                if 'block' in name.lower() or 'stage' in name.lower():
                    trainable_candidates.append(module)
            
            # Unfreeze the last N candidates
            for module in trainable_candidates[-self.num_to_train:]:
                for param in module.parameters():
                    param.requires_grad = True

        # 5. Trainable Head
        # Cellpose styles are typically 256. If using ViT, we detect the dim.
        self.fc = nn.Linear(256, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x.to(self.target_dtype)

        # 6. FORWARD PASS
        # Newer Cellpose nets return (mask, styles/latent)
        outputs = self.backbone(x)
        
        # Handle different return types (tuple vs tensor)
        if isinstance(outputs, (tuple, list)):
            styles = outputs[1]
        else:
            styles = outputs

        # Global pooling if the backbone returns spatial features
        if len(styles.shape) == 4:
            styles = styles.mean(dim=[2, 3])
            
        # Final classification
        return self.fc(styles.to(torch.float32))

class CellposeClassifier_Dynamic_Frozen(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        # 1. Setup metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_ch = args.sequence_length 
        self.requested_layers = args.num_layers 
        self.num_to_train = args.num_lastlayerstrain 
        
        # 2. Load Cellpose backbone
        # gpu=True in CellposeModel only controls internal CP logic; 
        # we still move the net to our device explicitly.
        cp_model = CellposeModel(gpu=torch.cuda.is_available())
        self.backbone = cp_model.net.to(self.device) 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 3. DYNAMIC INPUT LAYER REPLACEMENT
        # This replaces the initial projection (Conv2d) for 5-channel input
        self._replace_first_conv()

        # 4. IDENTIFY STAGES (SAM/Transformer vs UNet)
        if hasattr(self.backbone, 'blocks'):
            # Transformer path (Cellpose 3 / SAM)
            all_stages = list(self.backbone.blocks)
            self.is_transformer = True
        elif hasattr(self.backbone, 'downsample'):
            # Classic CNN path
            all_stages = list(self.backbone.downsample)
            self.is_transformer = False
        else:
            all_stages = list(self.backbone.children())
            self.is_transformer = False

        # 5. DYNAMIC SLICING & FREEZING
        limit = max(1, min(self.requested_layers, len(all_stages)))
        self.down_stages = nn.ModuleList(all_stages[:limit]).to(self.device)
        
        # Freeze everything first
        for param in self.down_stages.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers
        if self.num_to_train > 0:
            start_unfreeze = max(0, limit - self.num_to_train)
            for i in range(start_unfreeze, limit):
                for param in self.down_stages[i].parameters():
                    param.requires_grad = True

        # Always unfreeze the newly created input layer
        self._ensure_input_trainable()

        # 6. AUTO-DETECT Classifier Input Shape
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_ch, 256, 256).to(self.device).to(self.target_dtype)
            feat = self._forward_backbone(dummy_input)
            in_features = feat.shape[1]
        self.train()

        # 7. Classifier Head
        self.fc = nn.Linear(in_features, num_classes).to(self.device)

    def _replace_first_conv(self):
        """Iterates through the backbone to find the first Conv2d and update its channels."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels,
                    kernel_size=module.kernel_size, stride=module.stride,
                    padding=module.padding, bias=(module.bias is not None)
                ).to(self.device).to(self.target_dtype)
                
                # Update the module in the nested structure
                name_parts = name.rsplit('.', 1)
                if len(name_parts) > 1:
                    parent = dict(self.backbone.named_modules())[name_parts[0]]
                    setattr(parent, name_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                break

    def _ensure_input_trainable(self):
        """Finds the first layer in the sliced path and ensures it is trainable."""
        for param in self.down_stages[0].parameters():
            param.requires_grad = True
            break

    def _forward_backbone(self, x):
        """Architecture-aware forward pass that handles SAM's positional embeddings."""
        if self.is_transformer:
            # Step for Segment Anything (SAM) style encoders
            if hasattr(self.backbone, 'patch_embed'):
                x = self.backbone.patch_embed(x)
            if hasattr(self.backbone, 'pos_embed'):
                # Crucial Fix: Ensure pos_embed is on the same device as input x
                x = x + self.backbone.pos_embed.to(x.device)
            
        for stage in self.down_stages:
            x = stage(x)
        
        # Transform [B, Tokens, Dim] to [B, Dim, H, W] for pooling
        if len(x.shape) == 3: 
            x = x.transpose(1, 2)
            grid_size = int(x.shape[2]**0.5)
            x = x.reshape(x.shape[0], x.shape[1], grid_size, grid_size)
        return x

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # Cellpose/SAM expectation is 256x256
        if x.shape[2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        x = x.to(self.device).to(self.target_dtype)
        feat = self._forward_backbone(x)
        
        # Global Average Pooling
        pooled = feat.mean(dim=[2, 3])
        return self.fc(pooled.to(torch.float32))
class CellposeClassifier_Dynamic_TrainAll(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Load Cellpose backbone
        # We initialize the base CP network. By default, it uses a modified UNet.
        cp_model = CellposeModel(gpu=torch.cuda.is_available())
        self.backbone = cp_model.net 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 2. DYNAMIC INPUT LAYER REPLACEMENT
        # We replace the first Conv2d layer to accept 5 channels (the sequence length)
        self.first_layer_name = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None)
                ).to(self.target_dtype)
                
                # Locate parent and replace
                name_parts = name.rsplit('.', 1)
                if len(name_parts) > 1:
                    parent = dict(self.backbone.named_modules())[name_parts[0]]
                    setattr(parent, name_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                
                self.first_layer_name = name
                break

        # 3. ENABLE GRADIENTS FOR ALL (No Freezing)
        # Allows the entire morphological feature extractor to adapt
        for param in self.backbone.parameters():
            param.requires_grad = True

        # 4. Trainable Head
        # Cellpose 'style' vectors are typically 256-dimensional
        self.fc = nn.Linear(256, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        print(f"Cellpose Backbone: First layer ({self.first_layer_name}) updated. ALL layers TRAINING.")

    def forward(self, x):
        # Handle [B, 1, 5, H, W] -> [B, 5, H, W]
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)

        # Cellpose models are optimized for 256x256 internal resolution
        if x.shape[2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        x = x.to(self.target_dtype)

        # 5. FORWARD PASS
        # Cellpose nets usually return (mask, latent_styles)
        outputs = self.backbone(x)
        
        # Extract the style vector (global descriptor)
        if isinstance(outputs, (tuple, list)):
            # Index 1 is the 'style' vector in standard Cellpose UNets
            styles = outputs[1]
        else:
            styles = outputs

        # Global Average Pooling if the output is still spatial
        if len(styles.shape) == 4:
            styles = styles.mean(dim=[2, 3])
            
        # Final classification (ensure float32 for loss calculation)
        return self.fc(styles.to(torch.float32))
    
class CellposeClassifier_Frozenl(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()
        self.in_ch = args.sequence_length # 5
        
        # 1. Load Cellpose backbone (SAM/ViT version)
        cp_model = CellposeModel(gpu=torch.cuda.is_available(), model_type='cyto')
        self.backbone = cp_model.net 
        self.target_dtype = next(self.backbone.parameters()).dtype

        # 2. DYNAMICALLY REPLACE THE FIRST CONV LAYER
        self.first_layer_path = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                path_parts = name.rsplit('.', 1)
                new_layer = nn.Conv2d(
                    self.in_ch, module.out_channels, 
                    kernel_size=module.kernel_size, 
                    stride=module.stride, 
                    padding=module.padding,
                    bias=(module.bias is not None)
                ).to(self.target_dtype)
                
                if len(path_parts) > 1:
                    parent = dict(self.backbone.named_modules())[path_parts[0]]
                    setattr(parent, path_parts[1], new_layer)
                else:
                    self.backbone._modules[name] = new_layer
                
                self.first_layer_path = name
                break

        # 3. FREEZE EVERYTHING in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 4. UNFREEZE ONLY THE MODIFIED FIRST CONV
        if self.first_layer_path:
            # We target only the first layer we just created
            first_layer = dict(self.backbone.named_modules())[self.first_layer_path]
            for param in first_layer.parameters():
                param.requires_grad = True

        # 5. DEFINE & UNFREEZE THE FINAL FC LAYER
        # ViT-SAM outputs a 256-dimensional style vector
        self.fc = nn.Linear(256, num_classes)
        for param in self.fc.parameters():
            param.requires_grad = True

        # 6. VERIFICATION PRINT
        print("\n" + "="*30)
        print("TRAINABLE LAYERS SUMMARY:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"-> {name}")
        print("="*30 + "\n")

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)  
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.repeat(1, 5, 1, 1)

        # Match ViT-SAM patch grid (32x32 patches)
        # Based on your previous errors, 256x256 is the correct size.
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = x.to(self.target_dtype)

        # Forward pass: extract global morphological styles
        _, styles = self.backbone(x)
        
        if len(styles.shape) == 4:
            styles = styles.mean(dim=[2, 3])
            
        styles = styles.to(torch.float32)
        return self.fc(styles)
# MODEL REGISTRY
# -------------------------
MODEL_REGISTRY = {
    "C1_cnn2d": CNN2D,
    "C1_lenet5": LeNet5,

    "C1_alexnet": AlexNetClassifier,
    "C2_alexnet_Frozen":AlexNetClassifier_Frozen,
    "C4_alexnet_Dynamic_Frozen":AlexNetClassifier_Dynamic_Frozen,
    "C3_alexnet_Dynamic_TrainAll":AlexNetClassifier_Dynamic_TrainAll,

    "C1_vgg16": VGG16Classifier,
    "C2_vgg16_Frozen": VGG16Classifier_Frozen,
    "C4_vgg16_Dynamic_Frozen": VGG16Classifier_Dynamic_Frozen,
    "C3_vgg16_Dynamic_TrainAll": VGG16Classifier_Dynamic_TrainAll,

    "C1_vgg19": VGG19Classifier,
    "C2_vgg19_Frozen": VGG19Classifier_Frozen,    
    "C4_vgg19_Dynamic_Frozen":VGG19Classifier_Dynamic_Frozen,
    "C3_vgg19_Dynamic_TrainAll":VGG19Classifier_Dynamic_TrainAll,
    
    "C1_resnet18": ResNet18Classifier,
    "C2_resnet18_Frozen": ResNet18Classifier_Frozen,
    "C4_resnet18_Dynamic_Frozen": ResNet18Classifier_Dynamic_Frozen,  
    "C3_resnet18_Dynamic_TrainAll": ResNet18Classifier_Dynamic_TrainAll,

    "C1_resnet34": ResNet34Classifier,
    "C2_resnet34_Frozen": ResNet34Classifier_Frozen,
    "C4_resnet34_Dynamic_Frozen": ResNet34Classifier_Dynamic_Frozen, 
    "C3_resnet34_Dynamic_TrainAll": ResNet34Classifier_Dynamic_TrainAll,

    "C1_resnet50": ResNet50Classifier,
    "C2_resnet50_Frozen": ResNet50Classifier_Frozen,#"resnet50_Dynamic_Frozen": ResNet50_Dynamic_Frozen, 
    "C4_resnet50_Dynamic_Frozen": ResNet50Classifier_Dynamic_Frozen,#"resnet50_Dynamic_Frozen": ResNet50_Dynamic_Frozen, 
    "C3_resnet50_Dynamic_TrainAll": ResNet50Classifier_Dynamic_TrainAll,

    "C1_resnet101": ResNet101Classifier,
    "C2_resnet101_Frozen": ResNet101Classifier_Frozen,#"
    "C4_resnet101_Dynamic_Frozen": ResNet101Classifier_Dynamic_Frozen, 
    "C3_resnet101_Dynamic_TrainAll": ResNet101Classifier_Dynamic_TrainAll,

    "C1_densenet121": DenseNet121Classifier,
    "C2_densenet121_Frozen": DenseNet121Classifier_Frozen,
    "C4_densenet121_Dynamic_Frozen": DenseNet121Classifier_Dynamic_Frozen,
    "C3_densenet121_Dynamic_TrainAll": ResNet101Classifier_Dynamic_TrainAll,

    "C1_densenet169": DenseNet169Classifier,
    "C2_densenet169_Frozen": DenseNet169Classifier_Frozen,
    "C4_densenet169_Dynamic_Frozen": DenseNet169Classifier_Dynamic_Frozen,
    "C3_densenet169_Dynamic_TrainAll": DenseNet169Classifier_Dynamic_TrainAll,

    "C1_densenet201": DenseNet201Classifier,
    "C2_densenet201_Frozen": DenseNet201Classifier_Frozen,
    "C4_densenet201_Dynamic_Frozen": DenseNet201Classifier_Dynamic_Frozen,
    "C3_densenet201_Dynamic_TrainAll": DenseNet201Classifier_Dynamic_TrainAll,
    
    "C1_efficientnet_b0": EfficientNetB0Classifier,
    "C2_efficientnet_b0_Frozen": EfficientNetB0Classifier_Frozen,
    "C4_efficientnet_b0_Dynamic_Frozen": EfficientNetB0Classifier_Dynamic_Frozen,
    "C3_efficientnet_b0_Dynamic_TrainAll": EfficientNetB0Classifier_Dynamic_TrainAll,

    "C1_efficientnet_b1": EfficientNetB1Classifier,
    "C2_efficientnet_b1_Frozen": EfficientNetB1Classifier_Frozen,
    "C4_efficientnet_b1_Dynamic_Frozen": EfficientNetB1Classifier_Dynamic_Frozen,
    "C3_efficientnet_b1_Dynamic_TrainAll": EfficientNetB1Classifier_Dynamic_TrainAll,
    
    "C1_efficientnet_b2": EfficientNetB2Classifier,
    "C2_efficientnet_b2_Frozen": EfficientNetB2Classifier_Frozen,
    "C4_efficientnet_b2_Dynamic_Frozen": EfficientNetB1Classifier_Dynamic_Frozen,
    "C3_efficientnet_b2_Dynamic_TrainAll": EfficientNetB2Classifier_Dynamic_TrainAll,
    
    "C1_efficientnet_b3": EfficientNetB3Classifier,
    "C2_efficientnet_b3_Frozen": EfficientNetB3Classifier_Frozen,
    "C4_efficientnet_b3_Dynamic_Frozen": EfficientNetB3Classifier_Dynamic_Frozen,
    "C3_efficientnet_b3_Dynamic_TrainAll": EfficientNetB3Classifier_Dynamic_TrainAll,

    "C1_efficientnet_b4": EfficientNetB4Classifier,
    "C2_efficientnet_b4_Frozen": EfficientNetB4Classifier_Frozen,
    "C4_efficientnet_b4_Dynamic_Frozen": EfficientNetB4Classifier_Dynamic_Frozen,
    "C3_efficientnet_b4_Dynamic_TrainAll": EfficientNetB4Classifier_Dynamic_TrainAll,

    "C1_efficientnet_b5": EfficientNetB5Classifier,
    "C2_efficientnet_b5_Frozen": EfficientNetB5Classifier_Frozen,
    "efficientnet_b5_Dynamic_Frozen": EfficientNetB5Classifier_Dynamic_Frozen,
    "C3_efficientnet_b5_Dynamic_TrainAll": EfficientNetB5Classifier_Dynamic_TrainAll,

    "C1_efficientnet_b6": EfficientNetB6Classifier,
    "C2_efficientnet_b6_Frozen": EfficientNetB6Classifier_Frozen,
    "C4_efficientnet_b6_Dynamic_Frozen": EfficientNetB6Classifier_Dynamic_Frozen,
    "C3_efficientnet_b6_Dynamic_TrainAll": EfficientNetB6Classifier_Dynamic_TrainAll,

    "C1_efficientnet_b7": EfficientNetB7Classifier,
    "C2_efficientnet_b7_Frozen": EfficientNetB7Classifier_Frozen,
    "C4_efficientnet_b7_Dynamic_Frozen": EfficientNetB7Classifier_Dynamic_Frozen,
    "C3_efficientnet_b7_Dynamic_TrainAll": EfficientNetB7Classifier_Dynamic_TrainAll,

    "C1_unet": UNetClassifier,
    "C2_unet_Frozen": UNetClassifier_Frozen,
    "C4_unet_Dynamic_Frozen": UNetClassifier_Dynamic_Frozen,
    "C3_unet_Dynamic_TrainAll": UNetClassifier_Dynamic_TrainAll,

    "C1_unetpp": UNetPPClassifier,
    "C2_unetpp_Frozen": UNetPPClassifier_Frozen,
    "C4_unetpp_Dynamic_Frozen": UNetPPClassifier_Dynamic_Frozen,
    "C3_unetpp_Dynamic_TrainAll": UNetPPClassifier_Dynamic_TrainAll,
    
    "C1_mobilenet_v2": MobileNetV2Classifier,
    "C2_mobilenet_v2_Frozen": MobileNetV2Classifier_Frozen,
    "C4_mobilenet_v2_Dynamic_Frozen": MobileNetV2Classifier_Dynamic_Frozen,
    "C3_mobilenet_v2_Dynamic_TrainAll": MobileNetV2Classifier_Dynamic_TrainAll,
    

    "C1_mobilenet_v3_small": MobileNetV3SmallClassifier,
    "C2_mobilenet_v3_small_Frozen": MobileNetV3SmallClassifier_Frozen,
    "C4_mobilenet_v3_small_Dynamic_Frozen": MobileNetV3SmallClassifier_Dynamic_Frozen,
    "C3_mobilenet_v3_small_Dynamic_TrainAll": MobileNetV3SmallClassifier_Dynamic_TrainAll,
    
    "C1_mobilenet_v3_large": MobileNetV3LargeClassifier,
    "C2_mobilenet_v3_large_Frozen": MobileNetV3LargeClassifier_Frozen,
    "C4_mobilenet_v3_large_Dynamic_Frozen": MobileNetV3LargeClassifier_Dynamic_Frozen,
    "C3_mobilenet_v3_large_Dynamic_TrainAll": MobileNetV3LargeClassifier_Dynamic_TrainAll,

    "C1_vit": ViTClassifier,
    "C2_vit_Frozen":  ViTClassifier_Frozen,   
    "C4_vit_Dynamic_Frozen":  ViTClassifier_Dynamic_Frozen,
    "C3_vit_Dynamic_TrainAll":  ViTClassifier_Dynamic_TrainAll,

    "C1_swin": SwinClassifier,
    "C2_swin_Frozen": SwinClassifier_Frozen,
    "C4_swin_Dynamic_Frozen": SwinClassifier_Dynamic_Frozen,
    "C3_swin_Dynamic_TrainAll": SwinClassifier_Dynamic_TrainAll,

    "C1_attention_unet": AttentionUNetClassifier,
    "C2_attention_unet_Frozen": AttentionUNetClassifier_Frozen,
    "C4_attention_unet_Dynamic_Frozen": AttentionUNetClassifier_Dynamic_Frozen,
    "C3_attention_unet_Dynamic_TrainAll": AttentionUNetClassifier_Dynamic_TrainAll,
    

    "C1_deeplabv3": DeepLabV3Classifier,
    "C2_deeplabv3_Frozen": DeepLabV3Classifier_Frozen,
    "C2_deeplabv3_Dynamic_Frozen": DeepLabV3Classifier_Dynamic_Frozen,
    "C3_deeplabv3_Dynamic_TrainAll": DeepLabV3Classifier_Dynamic_TrainAll,

    "C1_hrnet": HRNetClassifier,
    "C2_hrnet_Frozen": HRNetClassifier_Frozen,
    "C4_hrnet_Dynamic_Frozen": HRNetClassifier_Dynamic_Frozen,
    "C3_hrnet_Dynamic_TrainAll": HRNetClassifier_Dynamic_TrainAll,

    "C1_segformer": SegFormerClassifier,
    "C2_segformer_Frozen": SegFormerClassifier_Frozen,
    "C4_segformer_Dynamic_Frozen": SegFormerClassifier_Dynamic_Frozen,
    "C3_segformer_Dynamic_TrainAll": SegFormerClassifier_Sliced_TrainAll,

    "C1_unet_resnet": UNetResNetClassifier,
    "C2_unet_resnet_Frozen": UNetResNetClassifier_Frozen,
    "C4_unet_resnet_Dynamic_Frozen": UNetResNetClassifier_Dynamic_Frozen,
    "C3_unet_resnet_Dynamic_TrainAll": UNetResNetClassifier_Dynamic_TrainAll,

    "C1_attention_unet_eff": AttentionUNetEfficientNet,
    "C2_attention_unet_eff_Frozen": AttentionUNetEfficientNet_Frozen,
    "C4_attention_unet_eff_Dynamic_Frozen": AttentionUNetEfficientNet_Dynamic_Frozen,
    "C3_attention_unet_eff_Dynamic_TrainAll": AttentionUNetEfficientNet_Dynamic_TrainAll, 
    
    "C1_cellpose": CellposeClassifier,
    "C2_cellpose_Frozen": CellposeClassifier_Frozen,
    "C4_cellpose_Dynamic_Frozen": CellposeClassifier_Dynamic_Frozen,
    "C3_cellpose_Dynamic_TrainAll": CellposeClassifier_Dynamic_TrainAll,
     
}

