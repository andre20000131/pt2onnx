#this UNet has no Depth2Space (PixelShuffle)
#using nearest upsamping...
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

#import arch_util as arch_util
import codes.models.modules.arch_util as arch_util


class HDRUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)

        #self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv1 = nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'))
        #self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        cond_in_nc=3
        cond_nf=64
        self.cond_first = nn.Sequential(nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
                                        nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True))
        self.CondNet1 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2d(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), 
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 3, 1, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, nf, 1),
                                      nn.ReLU(inplace=True), 
                                      nn.Conv2d(nf, out_nc, 1),
                                     )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # x[0]: img; x[1]: cond
        mask = self.mask_est(x)   #3,512,512 -> 64,512,512

        cond = self.cond_first(x)   #3,512,512 -> 64,512,512
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)
    
        fea0 = self.act(self.conv_first(x))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2

        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))

        out = self.act(self.up_conv1(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask * x + out
        return out

# if __name__ == '__main__':
#     input = torch.randn([1,3,512,512])
#     in2 = torch.randn([1,3,512,512])
#     a = in2
#     #a = torch.unsqueeze(a,0)
#     #print(a.shape)
#     #print(a[0].shape)
#     model = HDRUNet()
#     output = model(a)
#     print(output.shape)

if __name__ == '__main__':
    checkpoint = '../../pretrained_models/100w_new.pth'
    onnx_path = '../../../pretrained_models/100w_crop.onnx'
    input = torch.randn(1, 3, 512, 512)
    input_final = (input)
    input_names = ['input1']
    output_names = ['output']
    #pth_to_onnx(input_final, checkpoint, onnx_path, input_names, output_names)
    model_rel = HDRUNet()
    model = torch.load('../../../pretrained_models/100w_new.pth', map_location=torch.device('cpu'))
    # #指定模型的输入，以及onnx的输出路径
    model_rel.eval()
    model_rel.load_state_dict(model)
    torch.onnx.export(model_rel, input_final, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
    # 指定模型的输入，以及onnx的输出路径