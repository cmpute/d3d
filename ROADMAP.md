# TODOs

- [x] Port voxel generator
- [x] Port NMS cuda
- [ ] port in spconv and solve bugs: https://github.com/traveller59/spconv/issues/74 (also track https://github.com/poodarchu/Det3D/issues/71)
- [ ] Include visualization based on [pptk, pcl-py, open3d]
- [ ] Reimplement spconv with torch.sparse.XXXTensor and also support torch.is_contiguous

# Minor enhancements

- [ ] Make spconv available for pytorch 1.4+
- [ ] Include Nvidia/kaolin functions
- [ ] Include debugging and profiling tools: torchsnooper and snoop
  - Improvement: implement torchsnooper for SparseConvTensor
  - Improvement: let snoop output value if the tensor is a scalar
