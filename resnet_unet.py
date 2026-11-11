"""U-Net with an ImageNet-pretrained ResNet-34 encoder (torchvision) and a U-Net decoder.

Encoder outputs (for a 512x512 input):
    stem   (conv1+bn+relu)  64 ch, 256x256   (1/2)
    layer1                  64 ch, 128x128   (1/4)
    layer2                 128 ch,  64x64    (1/8)
    layer3                 256 ch,  32x32    (1/16)
    layer4                 512 ch,  16x16    (1/32)
The decoder upsamples step by step, joining each encoder output through a skip connection,
then a last upsampling step brings it back to full resolution.
"""
import torch
import torchvision
import unet
from unet import ConvBlock

#official torchvision ImageNet weights; the hash in the file name is checked after download
RESNET34_URL="https://download.pytorch.org/models/resnet34-b627a593.pth"
PRETRAINED_DIR="pretrained"   #inside the project (gitignored), not ~/.cache


class DecoderBlock(torch.nn.Module):
    """Upsample x2, join the skip connection (if any), then two 3x3 convs."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up=torch.nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv=ConvBlock(in_channels+skip_channels,out_channels)

    def forward(self, x, skip=None):
        x=self.up(x)
        if skip is not None:
            x=torch.cat([x,skip],dim=1)
        return self.conv(x)


class ResNet34UNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        resnet=torchvision.models.resnet34(weights=None)
        if pretrained:
            state=torch.hub.load_state_dict_from_url(RESNET34_URL,model_dir=PRETRAINED_DIR,
                                                     map_location="cpu",check_hash=True,progress=True)
            resnet.load_state_dict(state)

        #first conv expects 3 colour channels: for greyscale input, add the three colour filters together
        #(the same as feeding the grey image into all three channels)
        if in_channels!=3:
            old=resnet.conv1
            resnet.conv1=torch.nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
            with torch.no_grad():
                summed=old.weight.sum(dim=1,keepdim=True)              #(64,1,7,7)
                resnet.conv1.weight.copy_(summed.repeat(1,in_channels,1,1)/in_channels)

        #encoder: ResNet-34 without its classification head (avgpool + fc)
        self.stem=torch.nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu)   #1/2
        self.pool=resnet.maxpool                                              #1/4
        self.layer1=resnet.layer1    #64 ch,  1/4
        self.layer2=resnet.layer2    #128 ch, 1/8
        self.layer3=resnet.layer3    #256 ch, 1/16
        self.layer4=resnet.layer4    #512 ch, 1/32

        #decoder
        self.dec4=DecoderBlock(512,256,256)   #1/32 -> 1/16
        self.dec3=DecoderBlock(256,128,128)   #1/16 -> 1/8
        self.dec2=DecoderBlock(128,64,64)     #1/8  -> 1/4
        self.dec1=DecoderBlock(64,64,64)      #1/4  -> 1/2 (joins the stem)
        self.dec0=DecoderBlock(64,0,32)       #1/2  -> full size (no skip at full resolution)
        self.head=torch.nn.Conv2d(32,out_channels,kernel_size=1)

    def forward(self, x):
        if x.shape[-2]%32 or x.shape[-1]%32:
            raise ValueError(f"Input height and width must be divisible by 32, got {tuple(x.shape[-2:])}")
        s=self.stem(x)
        e1=self.layer1(self.pool(s))
        e2=self.layer2(e1)
        e3=self.layer3(e2)
        e4=self.layer4(e3)
        d=self.dec4(e4,e3)
        d=self.dec3(d,e2)
        d=self.dec2(d,e1)
        d=self.dec1(d,s)
        d=self.dec0(d)
        return torch.sigmoid(self.head(d))


def build_model(name="unet", pretrained=True):
    """Model by name, so main.py and evaluate.py build the same architecture from the run config."""
    if name=="unet":
        return unet.UNet(in_channels=1,out_channels=1)
    if name=="resnet34_unet":
        return ResNet34UNet(in_channels=1,out_channels=1,pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'")
