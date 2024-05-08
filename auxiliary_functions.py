import torch
import numpy as np
import cv2 as cv
import os
import math
import pil as Image # Should change to cv2 just for consistency

def median_filter(path, blur_radius = 6):
    """
    Args:
        path (string): Path to the folder containing the images to be filtered.
        blur_radius (int, optional): Blur radius for the median filter. Defaults to 6.

    Returns:
        None: All information gets saved in the path + 'median' folder.
    """
    
    # Read images
    accepted_filetypes = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp']
    
    try:
        os.mkdir(path+'median')
    except OSError as error:
        pass
    
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1][1:] # Get file extension
        if ext.lower() not in accepted_filetypes:
            continue
        cv.imwrite(path+'median/'+f+'.tiff' ,(cv.imread(path+f, cv.IMREAD_GRAYSCALE).astype(np.float32)-cv.medianBlur(cv.imread(path+f, cv.IMREAD_GRAYSCALE).astype(np.float32),blur_radius)))
    
    return None

def img_to_pt(path):
    '''Converts all images in a folder to a pyTorch tensor.
    Args:
        path (string): Path to the folder containing the images.
        
    Returns:
        tensor_imgs (torch.Tensor): PyTorch Tensor containing all the images.
    '''
    # Find all images in a folder:
    imgs = []
    accepted_filetypes = ['tif', 'tiff', 'png', 'jpg', 'jpeg', 'bmp']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1][1:]
        if ext.lower() not in accepted_filetypes:
            continue
        imgs.append(cv.imread(path+f, cv.IMREAD_UNCHANGED).astype(np.float32))

    # Convert them into a tensor for pyTorch
    tensor_imgs = torch.from_numpy(np.array(imgs,dtype=np.float32)).to(torch.float32)
    del imgs
    gc.collect()
    
    return tensor_imgs

def pt_to_img(path):
    # Find all images in a folder:
    imgs = []
    accepted_filetypes = ['pt']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1][1:]
        if ext.lower() not in accepted_filetypes:
            continue
        for i in range(len(torch.load(path+f, map_location=torch.device('cpu')))):
            cv.imwrite(path+f+'_'+str(i)+'.tif', torch.load(path+f, map_location=torch.device('cpu'))[i,:,:].numpy())
    return None

def two_n_squareify(tensor):
    '''
    Args:
        tensor (torch.Tensor): Input tensor to be have its dimensions rounded up to the nearest 2^n. Should be in the format of [depth, x, y].
    Returns:
        return_tensor (torch.Tensor): Tensor with its dimensions rounded up to the nearest 2^n. Image will first be croped to 2^n-1/2^n-1, then padded with zeros to 2^n/2^n. This is to ensure linear convolution in the Fourier domain.
    '''
    # This function takes a tensor and pads it with zeros until the dimensions are a power of 2
    # The return tensor is a square tensor

    old_shape = tensor.shape

    #check and make sure .pt is in format of (depth, x, y)
    if len(tensor.shape) == 3:
        if tensor.shape[-1] > tensor.shape[0]:
            pass
        else:
            # this means .pt is in format of (x, y, depth)
            tensor = tensor.permute(2,0,1)
            print('Transposed tensor to (depth, x, y) format', str(tensor.shape))
        
    
    # Find the next power of 2
    next_power = max(2**math.ceil(math.log2(len(tensor[0,:,0]))), 2**math.ceil(math.log2(len(tensor[0,0,:]))))

    # Check if the tensor is already a power of 2
    if np.log2(len(tensor[0,:,0])).is_integer:
        next_power = int(2 * len(tensor[0,:,0]))
        pad = int(next_power / 4)
        return_tensor = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), "constant", 0)

    else: 
    
        # Find the amount of padding needed
        x_pad = next_power - len(tensor[0,:,0])
        y_pad = next_power - len(tensor[0,0,:])

        # Check to see if it's even or odd
        if x_pad % 2 == 0:
            l_x_pad = int(x_pad/2)
            r_x_pad = int(x_pad/2)
        else:
            l_x_pad = int(x_pad/2)
            r_x_pad = x_pad - int(x_pad/2)

        if y_pad % 2 == 0:
            l_y_pad = int(y_pad/2)
            r_y_pad = int(y_pad/2)
        else:
            l_y_pad = int(y_pad/2)
            r_y_pad = y_pad - int(y_pad/2)

        mask_tensor = torch.ones(len(tensor[:,0,0]), next_power, next_power)

        mask_tensor[:,0:int(next_power/4),:] = 0
        mask_tensor[:,:,0:int(next_power/4)] = 0
        mask_tensor[:,int(3*next_power/4):,:] = 0
        mask_tensor[:,:,int(3*next_power/4):] = 0

        return_tensor = torch.mul(torch.nn.functional.pad(tensor, (l_y_pad, r_y_pad, l_x_pad, r_x_pad), "constant", 0), mask_tensor)
    
    new_shape = return_tensor.shape

    print('Old shape: '+ str(old_shape) +', '+ ' New shape: ' + str(new_shape))

    return return_tensor

def load_psf(path, normalization_type = None):
    # Loads the PSF stack from a path
    #path = r'/Users/halensolomon/Code/FLFM_local/testing/psf_stack.pt'
    
    # if the file type is .pt, then we can just load it
    if path[-3:] == '.pt':
        psf = torch.load(path)
        if normalization_type == 'forward':
            psf = torch.div(psf, torch.sum(psf, dim = (0))) # Sum of each axial slice should be 1
        if normalization_type == 'backward':
            psf = torch.div(psf, torch.sum(psf, dim = (1,2))) # Sum of each axial slice should be 1
    
    # if the path is actually a folder, then we need to load in all the files in the folder
    elif os.path.isdir(path):
        psf = torch.zeros(len(os.listdir(path)), len(Image.open(path + '/' + os.listdir(path)[0]).convert('L').getdata()), len(Image.open(path + '/' + os.listdir(path)[0]).convert('L').getdata()))
        for i in range(len(os.listdir(path))): # Format is [depth, x, y]
            psf[i,:,:] = torch.from_numpy(np.array(Image.open(path + '/' + os.listdir(path)[i]).convert('L').getdata()).reshape(Image.open(path + '/' + os.listdir(path)[i]).convert('L').size))
        if normalization_type == 'forward':
            psf = torch.div(psf, torch.sum(psf, dim = (0)))
        if normalization_type == 'backward':
            psf = torch.div(psf, torch.sum(psf, dim = (1,2)))
        
    return psf

def fov_cut(PSF, fov_radius):
    def circ(x, y, r):
        return (x**2 + y**2 < r**2).to(int)
    
    x = torch.range(0, len(PSF[0,0,:])-1)
    y = torch.range(0, len(PSF[0,:,0])-1)
    
    PSF_return = torch.empty(len(PSF[:,0,0]), len(PSF[0,0,:]), len(PSF[0,:,0]))
    
    for i in range(len(PSF[:,0,0])):
        PSF_return[i,:,:] = torch.mul(PSF[i,:,:], circ(x, y, fov_radius))
    
    return PSF_return

def f_b_norm(forward, backward, path):
    '''
    Forward: [depth, x, y], where depth is the same as the backward model
    Backward: [depth, x, y], where depth is the same as the forward model
    Path: The path to save the normalized forward and backward models (ex. r'/Users/yourname/yourfolder/)
    '''
    
    # Normalize so that the each forward depth slice sums to 1
    forward[forward <= 1e-8] = 1e-8 # Computational trick to avoid dividing by zero
    forward[:,:,:] = torch.div(forward[:,:,:], torch.sum(forward, dim = (1,2)).unsqueeze(1).unsqueeze(2)) # Brodcast the sum of the forward model to the same shape as the forward model
    forward[forward == 1/len(forward[:,0,0])] = 0 # Set the values that had no light to zero
    
    # Normalize so that each voxel for the backward model sums to 1
    backward[backward <= 1e-8] = 1e-8 # Computational trick to avoid dividing by zero
    backward[:,:,:] = torch.div(backward[:,:,:], torch.sum(backward, dim = (0)).unsqueeze(0)) # Brodcast the sum of the backward model to the same shape as the backward model
    backward[backward == 1/len(backward[:,0,0])] = 0 # Set the values that had no light to zero
    
    # Save the normalized forward and backward models
    torch.save(forward, path + "forward_norm.pt")
    torch.save(backward, path + "backward_norm.pt")
    
    return None # This function is only used to save the normalized forward and backward models

def mat_to_torch(file):
    '''
    Args:
        file (string): Path to the .mat file to be converted to a torch tensor.
    Returns:
        torch_tensor (torch.Tensor): PyTorch tensor containing the data from the .mat file.
    '''
    sparse_matrix = sio.loadmat(file, struct_as_record=False)
    sparse_matrix = sparse_matrix[list(sparse_matrix.keys())[-1]]
    torch_tensor = torch.empty(len(sparse_matrix[0,0,:]), sparse_matrix[0,0,0].shape[0], sparse_matrix[0,0,0].shape[1])
    
    for i in range(len(sparse_matrix[0,0,:])):
        temp_matrix = sparse_matrix[0,0,i].toarray()
        torch_tensor[i, :, :] = torch.from_numpy(temp_matrix)
    
    return torch_tensor

def image_rescaler(size, tensor):
    # Assuming the tensor is in the format of [depth, x, y]
    rescaled_tensor = tv.transforms.Resize(size = size)(tensor[:,:,:])
    return rescaled_tensor

def pt2tif(file, path):
    tens = torch.load(file).to(device = 'cpu')
    for i in range(len(tens[:,0,0])):
        cv.imwrite(path + str(i) + '.tif', tens[i,:,:].numpy())