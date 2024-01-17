lq_tensor = torch.tensor(lq_face.transpose(2,0,1))/255*2-1
lq_tensor = lq_tensor.unsqueeze(0).float().to(model.device)

print(lq_tensor.shape)
output_SR = model.netG(lq_tensor)
remove_all_spectral_norm(model.netG)
input_names = ['input',]
output_names = ['output',]
onnx_path ="./a.onnx"
torch.onnx.export(model.netG.module,lq_tensor,onnx_path,verbose=True,input_names=input_names,output_names=output_names)
