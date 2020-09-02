# TODOs
- [ ] Fix Nuscenes and Waymo image projection error (add motion compensation?)
- [ ] Fix Kitti detection output and add output for nuscenes
- [ ] port in https://github.com/poodarchu/Det3D
- [ ] Include visualization based on [pptk (nvidia/kaolin has example), open3d]
- [ ] Implement functions as torch script operator? First need to make output stored in function output

Some repos for reference:
- https://github.com/nicolas-chaulet/torch-points3d

# Minor enhancements

- [x] Make spconv available for pytorch 1.4+
- [ ] Include debugging and profiling tools: torchsnooper and snoop
  - Improvement: implement torchsnooper for SparseConvTensor
  - Improvement: let snoop output value if the tensor is a scalar
- [ ] Migrate all path processing to pathlib
