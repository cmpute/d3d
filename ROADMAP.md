# TODOs

- [x] Port voxel generator
- [x] Port NMS cuda
- [ ] port in spconv and solve bugs: https://github.com/traveller59/spconv/issues/74 (also track https://github.com/poodarchu/Det3D/issues/71)
- [ ] port in https://github.com/Oldpan/Pytorch-Memory-Utils
- [ ] port in https://github.com/poodarchu/Det3D
- [ ] Include visualization based on [pptk, pcl-py, open3d]
- [ ] Implement functions as torch script operator? First need to make output stored in function output
- [ ] Reimplement spconv with torch.sparse.XXXTensor and also support torch.is_contiguous / use MinkowskiEngine
- [ ] Implement torch_sparse.coalesce for normal sparse tensor...

# Minor enhancements

- [x] Make spconv available for pytorch 1.4+
- [ ] Include Nvidia/kaolin functions
- [ ] Include debugging and profiling tools: torchsnooper and snoop
  - Improvement: implement torchsnooper for SparseConvTensor
  - Improvement: let snoop output value if the tensor is a scalar
