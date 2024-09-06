import imageio
from PSF import create_psf, psf_cal, psf
import numpy as np

def save_psf(dxy: list, dz: list, SizeXY: list, SizeZ: list, lamd: list, NA: list, RI: list, filename: str):
    kernels = []
    for i in range(len(dxy)):
        # 根据参数生成 psf 图片
        psf_raw = create_psf(dxy[i], dz[i], SizeXY[i], SizeZ[i], lamd[i], NA[i], RI[i], filename=None) 
        # 根据 psf 图片计算 psf 卷积核
        # 未指定 index 深度默认为 psf 最明显的 SizeZ // 2
        kernels.append(psf_cal(psf_raw, index=None))
    kernels = np.stack(kernels)
    np.save(filename, kernels)
            
if __name__ == "__main__":

    dxy = [92.6, 92.6]  # lateral sampling, in (nm)
    dz = [92.6, 92.6]   # axial sampling, in (nm)
    SizeXY = [27, 27]  # lateral pixel number of PSF
    SizeZ = [13, 13]  # axial pixel number of PSF
    lamd = [525, 525]   # emission wavelength, in (nm)
    NA = [1.1, 1.1]     # numerical aperture
    RI = [1.3, 1.3]    # refractive index

    # 保存为 index, height, width, 1, 1 
    save_psf(dxy, dz, SizeXY, SizeZ, lamd, NA, RI, 'kernels.npy')

    loaded_array = np.load('kernels.npy')

    input = np.float32(imageio.imread("./im1_GT.tif"))
    output = psf(loaded_array[0], input)

    # 打印 output 图片
    output_np = output.squeeze().detach().numpy()
    output_np = np.clip(output_np, 0, 255)  
    output_np = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
    imageio.imwrite("./output.tif", output_np)
    