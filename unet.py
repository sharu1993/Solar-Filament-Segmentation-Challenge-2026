import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels,out_channels,kernel_size,padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)

class UNet(torch.nn.Module):
    def __init__(self,in_channels=3, out_channels=3,features=[64,128,256,512]):
        super(UNet,self).__init__()

        self.encoder=torch.nn.ModuleList()
        self.decoder=torch.nn.ModuleList()
        self.maxpool=torch.nn.MaxPool2d(kernel_size=2,stride=2)

        #encoder path
        self.encoder.append(ConvBlock(in_channels,features[0]))
        for i in range(1,len(features)):
            self.encoder.append(ConvBlock(features[i-1],features[i]))

        #bottleneck
        self.bottleneck=ConvBlock(features[-1],features[-1]*2)

        #decoder path (each step involes an upsampling then a convolution)
        self.decoder.append(torch.nn.ConvTranspose2d(features[-1]*2,features[-1],kernel_size=2,stride=2))
        self.decoder.append(ConvBlock(features[-1]*2,features[-1]))
        for i in range(len(features)-1,0,-1):
            self.decoder.append(torch.nn.ConvTranspose2d(features[i],features[i-1],kernel_size=2,stride=2))
            self.decoder.append(ConvBlock(features[i-1]*2,features[i-1]))

        #output layer
        self.final_conv=torch.nn.Conv2d(features[0],out_channels,kernel_size=1)
        

    def forward(self,x):
        #height/width must be divisible by 2**(number of poolings) so encoder and decoder sizes line up
        factor=2**len(self.encoder)
        if x.shape[-2]%factor or x.shape[-1]%factor:
            raise ValueError(f"Input size {tuple(x.shape[-2:])} must be divisible by {factor}")

        skip_connections=[]

        for encoder in self.encoder:
            x=encoder(x)
            skip_connections.append(x) #store output of encoders for skip connections. Will be used to augment decoder inputs
            x=self.maxpool(x)

        x=self.bottleneck(x)

        skip_connections=skip_connections[::-1] #reverse it

        for idx in range(0,len(self.decoder),2):
            x=self.decoder[idx](x)
            skip=skip_connections[idx//2]
            concat=torch.cat((skip,x),dim=1)
            x=self.decoder[idx+1](concat)

        # output layer
        x=self.final_conv(x)
        x=torch.sigmoid(x) #sets values between 0 and 1. threshold for binary mask?
        return x
