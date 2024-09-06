import numpy as np
from scipy.fft import fft2, ifftshift, fftshift
import tifffile

def create_psf(dxy, dz, SizeXY, SizeZ, lamd, NA, RI, filename = None):
    '''
        创建 psf \n
        dxy:    lateral sampling, in (nm) \n
        dz:     axial sampling, in (nm) \n
        SizeXY: lateral pixel number of PSF \n
        SizeZ:  axial pixel number of PSF \n
        lamd:   emission wavelength, in (nm) \n
        NA:     numerical aperture \n
        RI:     refractive index \n
    '''
    dk = 2 * np.pi / dxy / SizeXY
    kx = np.arange(-(SizeXY - 1) / 2, (SizeXY - 1) / 2 + 1) * dk
    kx, ky = np.meshgrid(kx, kx)
    kr_sq = kx**2 + ky**2
    z = np.arange(-(SizeZ - 1) / 2, (SizeZ - 1) / 2 + 1) * dz

    PupilMask = (kr_sq <= (2 * np.pi / lamd * NA)**2).astype(int)
    kz = np.sqrt(((2 * np.pi / lamd * RI)**2 - kr_sq).astype(np.complex128)) * PupilMask
    PSF = np.zeros((SizeXY, SizeXY, SizeZ), dtype=np.complex128)

    for ii in range(SizeZ):
        tmp = PupilMask * np.exp(1j * kz * z[ii])
        tmp = fftshift(fft2(ifftshift(tmp)))
        PSF[:, :, ii] = (np.abs(tmp))**2

    PSF = PSF / np.max(PSF) * 2**15
    PSF = PSF.astype(np.uint16)

    if filename is not None:
        with tifffile.TiffWriter(filename) as tiff:
            for i in range(SizeZ):
                tiff.write(PSF[:, :, i])

    return PSF