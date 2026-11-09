import torch
import pdb

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size,padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels,out_channels,kernel_size,padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)

class UNet(torch.nn.Module):
    def __init__(self,in_channels=3, out_channels=3,features=[64,126,256,512]):
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
        print(x.shape)
        skip_connections=[]

        for encoder in self.encoder:
            x=encoder(x)
            skip_connections.append(x) #store output of encoders for skip connections. Will be used to augment decoder inputs
            x=self.maxpool(x)
        print(x.shape)

        x=self.bottleneck(x)
        print(x.shape)

        skip_connections=skip_connections[::-1] #reverse it
        print(x.shape)

        pdb.set_trace()
        for idx in range(0,len(self.decoder),2):
            x=self.decoder[idx](x)
            skip=skip_connections[idx//2]
            if x.shape!=skip.shape: #handle size mismatch
                x=torch.nn.functional.interpolate(x,size=skip.shape[2:])
            concat=torch.cat((skip,x),dim=1)
            x=self.decoder[idx+1](concat)
            print(x.shape)
        pdb.set_trace()
        print(x.shape)

        # output layer
        x=self.final_conv(x)
        x=torch.sigmoid(x) #sets values between 0 and 1. threshold for binary mask?
        print(x.shape)
        return x

