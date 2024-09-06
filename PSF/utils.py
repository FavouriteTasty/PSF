import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def gaussian_1d(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def psf_estimator_2d(psf):
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    # estimate y sigma
    x = np.asarray(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x]))
    fit_y, cov_y, *_ = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2], full_output=True)
    print('estimated psf sigma_y: ', fit_y[2])
    # estimate x sigma
    x = np.asarray(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :]))
    fit_x, cov_x, *_ = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2], full_output=True)
    print('estimated psf sigma_x: ', fit_x[2])
    return fit_y[2], fit_x[2]

def psf_cal(psf, index = None):
    psf_g = np.float32(psf)
    if index is not None:
       psf_g = psf_g[index]
    else:
       psf_g = psf_g[psf_g.shape[0]//2] # 指定了深度
    psf_width,psf_height = psf_g.shape
    half_psf_width = psf_width//2

    dxypsf = 0.0313/2
    dx = 0.0313/2
    input_x = 128
    input_y = 128
    upsample_flag = 1

    # get the desired dxy
    if psf_width%2==1:
        sr_ratio = dxypsf/dx
        sr_x = round(psf_width*sr_ratio)
        if sr_x%2==0:
            if sr_x>psf_width*sr_ratio:
                sr_x = sr_x - 1
            else:
                sr_x = sr_x + 1
        sr_y = round(psf_height*sr_ratio)
        if sr_y%2==0:
            if sr_y>psf_height*sr_ratio:
                sr_y = sr_y - 1
            else:
                sr_y = sr_y + 1
        psf_g = cv2.resize(psf_g,(sr_x,sr_y))
    else:
        x = np.arange((half_psf_width+1) * dxypsf, (psf_width+0.1) * dxypsf, dxypsf)
        xi = np.arange((half_psf_width+1) * dxypsf, (psf_width+0.1) * dxypsf, dx)
        if xi[-1]>x[-1]:
            xi = xi[0:-1]
        PSF1 = np.zeros((len(xi),psf_height))
        for i in range(psf_height):
            curCol = psf_g[half_psf_width:psf_width,i]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[:,i] = interp(xi)
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x)):
            x2[len(x)-n-1]=x[0]-dxypsf*n
        for n in range(len(xi)):
            xi2[len(xi)-n-1]=xi[0]-dx*n
        PSF2 = np.zeros((len(xi2),psf_height))
        for i in range(psf_height):
            curCol = psf_g[1:half_psf_width+1+psf_width%2,i]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[:,i] = interp(xi2)
        psf_g = np.concatenate((PSF2[:-1,:],PSF1),axis=0)
        psf_g = psf_g/np.sum(psf_g)
        psf_width,psf_height = psf_g.shape
        half_psf_height = psf_height//2
        
        x = np.arange((half_psf_height+1) * dxypsf, (psf_height+0.1) * dxypsf, dxypsf)
        xi = np.arange((half_psf_height+1) * dxypsf, (psf_height+0.1) * dxypsf, dx)
        if xi[-1]>x[-1]:
            xi = xi[0:-1]
        PSF1 = np.zeros((psf_width,len(xi)))
        for i in range(psf_width):
            curCol = psf_g[i,half_psf_height:psf_height]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[i,:] = interp(xi)
        x2 = np.zeros(len(x))
        xi2 = np.zeros(len(xi))
        for n in range(len(x2)):
            x2[len(x2)-n-1]=x[0]-dxypsf*n
        for n in range(len(xi2)):
            xi2[len(xi2)-n-1]=xi[0]-dx*n
        PSF2 = np.zeros((psf_width,len(xi2)))
        for i in range(psf_width):
            curCol = psf_g[i,1:half_psf_height+1+psf_height%2]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[i,:] = interp(xi2)
        psf_g = np.concatenate((PSF2[:,:-1],PSF1),axis=1)
        
    otf_g = np.fft.fftshift(np.fft.fftn(psf_g))
    otf_g = np.abs(otf_g)
    otf_g = cv2.resize(otf_g,(input_x*(1+upsample_flag),input_y*(1+upsample_flag)))
    otf_g = otf_g/np.sum(otf_g)     
    psf_width = psf_g.shape[0]
    psf_height = psf_g.shape[1]

    # crop PSF for faster computation
    sigma_y, sigma_x = psf_estimator_2d(psf_g)
    ksize = int(sigma_y * 4)
    halfx = psf_width // 2
    halfy = psf_height // 2
    if ksize<=halfx:
        psf_g = psf_g[halfx-ksize:halfx+ksize+1, halfy-ksize:halfy+ksize+1]
        psf_g = np.reshape(psf_g,(2*ksize+1,2*ksize+1,1,1)).astype(np.float32)
    else:
        psf_g = np.reshape(psf_g,(psf_width,psf_height,1,1)).astype(np.float32)
    psf_g = psf_g/np.sum(psf_g)
        
    # save
    # psf_g_tosave = np.uint16(65535*prctile_norm(np.squeeze(psf_g)))
    # imageio.imwrite('psf.tif',psf_g_tosave)
    # otf_g_tosave = np.uint16(65535*prctile_norm(np.squeeze(np.abs(otf_g))))
    # imageio.imwrite('otf.tif',otf_g_tosave)

    return psf_g

def psf(psf_g, input):
   
    psf_g_tensor = torch.tensor(psf_g).permute(2, 3, 0, 1)
    input_tensor = torch.tensor(input).permute(0, 1).unsqueeze(0).unsqueeze(0)
    padding = (psf_g_tensor.shape[2] // 2, psf_g_tensor.shape[3] // 2)
    output = F.conv2d(input_tensor, psf_g_tensor, padding=padding, groups=input_tensor.shape[1])
    _, _, height, width = output.shape
    output = F.interpolate(output, size=(height//2, width//2), mode='bilinear', align_corners=False)
    return output