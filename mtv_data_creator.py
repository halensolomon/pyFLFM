import torch
import math
import numpy as np
import cv2 as cv
import os
import torchvision as tv
import gc

def gnd_ref_generator(itr, gnd_dir, img_dir, image_size, psf, cent, fov_rad, beamlet_rad, beamlet_space, beamlet_x, beamlet_y):
    '''
    Creates ground truth images and reference images for hypothetical MTV data.

    Parameters:
    itr: int
        Number of images to generate
    gnd_dir: str
        Path to the directory to store the ground truth images
    img_dir: str
        Path to the directory to store the reference images
    image_size: tuple
        Size of the image (height, width) in pixels
    psf: torch.Tensor
        Point spread function in (depth, height, width) format
    cent: tuple
        Center of the circular region of interest (in pixels)
    fov_rad: int
        Radius of the circular region of interest (in pixels)
    beamlet_rad: int
        Radius of the beamlet (in pixels)
    beamlet_space: int
        Spacing between the beamlets (in pixels)
    beamlet_x: int
        Number of beamlets in the x-direction
    beamlet_y: int
        Number of beamlets in the y-direction
    '''
    def circ(x,y,r):
        return ((x**2 + y**2) <= r**2).int()
    
    # Generate the velocity profile
    def one_seventh(U_inf, delta, y):
        '''Blausius velocity profile'''
        return (U_inf * 8.7 * ((y/delta))**(1/7))
    
    def convfft(image,kernel):
        # Assumes that the image and the kernel are the same size
        # Assumes that the image and the kernel are square
        # Assumes that the image and the kernel have been 2nified.
        
        image_fft = torch.fft.fft2(image.to(torch.float32), norm = 'backward')
        kernel_fft = torch.fft.fft2(kernel.to(torch.float32), norm = 'backward')
        fft_output = torch.mul(image_fft, kernel_fft)
        
        result = torch.fft.ifftshift(torch.abs(torch.fft.ifft2(fft_output, norm= 'backward'))).to(torch.float32)
        
        # Set edges to zero
        result[0:int(kernel.shape[0]/4), :] = result[int(3*kernel.shape[0]/4):, :] = result[:, :int(kernel.shape[1]/4)] = result[:, int(3*kernel.shape[1]/4):] = 0
        return result
    
    x,y = torch.meshgrid(torch.arange(int(image_size[0])), torch.arange(int(image_size[1])), indexing='xy')
    
    # Random number generator
    gen = torch.Generator() # Create a generator object
    gen.manual_seed(1989) # Set the seed for the random number generator, ensures reproducibility
    
    # Create paths to store different images at different depths
    if not os.path.exists(gnd_dir):
        os.mkdir(gnd_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for i in range(psf.shape[0]):
        if not os.path.exists(gnd_dir + '/depth' + str(i)):
            os.mkdir(gnd_dir + '/depth' + str(i))
    
    for a in range(itr):
        center = cent + (torch.randint(low=-20, high=20, size=(1,), generator = gen), torch.randint(low=-20, high=20, size=(1,), generator = gen)) # Add randomness to the center
        fov_radius = fov_rad + torch.randint(low=-50, high=50, size=(1,), generator = gen) # Add randomness to the radius
        depth_slices = psf.shape[0] # psf should be in the shape of (depth, height, width)
        
        beamlet_radius = int(beamlet_rad + torch.randint(low=-5, high=5, size=(1,), generator = gen)) # Add randomness to the beamlet radius
        beamlet_spacing = int(beamlet_space + torch.randint(low=-50, high=50, size=(1,), generator = gen)) # Add randomness to the beamlet spacing
        
        beamlet_centers = torch.from_numpy(np.array([(int(i*beamlet_spacing + center[0] - (((beamlet_x-1)//2)*beamlet_spacing)), int(j*beamlet_spacing + center[1] - (((beamlet_y-1)//2)*beamlet_spacing))) for i in range(beamlet_x) for j in range(beamlet_y)])) # 4x4 beamlets, equally spaced
        
        # Create random velocity profile
        U_inf = torch.abs(0.1 * torch.randn(size = (1,), generator = gen)) # Free stream velocity
        delta = torch.abs(5 * torch.randn(size = (1,), generator = gen)) # Boundary layer thickness
        velocity = torch.tensor([one_seventh(U_inf, delta, i) for i in range(depth_slices)])
        
        # Create circular beamlets for each beamlet center, with displacement from the center
        template = torch.zeros(depth_slices, image_size[0], image_size[1]) # ground truth image
        rand_angle = torch.tensor(math.pi * 2) * torch.abs(torch.randn(size = (1,), generator = gen)) # Random angle to rotate the velocity profile
        velocity_angled = torch.mul(velocity.unsqueeze(1), torch.tile(torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)]), (depth_slices, 1))) # Rotate the velocity profile
        
        # Create the beamlets
        for c in beamlet_centers:
            for i in range(template.shape[0]):
                template[i,:,:] = torch.add(template[i,:,:], circ(x-c[0]-velocity_angled[i,0], y-c[1]-velocity_angled[i,1], beamlet_radius)) # Add the beamlet to the ground truth image

        del beamlet_centers, beamlet_radius, velocity, velocity_angled, rand_angle, U_inf, delta  # Delete the variables to free up memory
        gc.collect() # Force garbage collection
        
        add_background = template + (0.1 * torch.rand_like(template)) # Add some background noise for the reference image
        template = torch.mul(template, circ(x-center[0], y-center[1], fov_radius)) # Only keep the circular region of interest 
        add_background = torch.mul(add_background, circ(x-center[0], y-center[1], fov_radius)) # Only keep the circular region of interest
        
        ref = torch.zeros(image_size)
        
        for i in range(depth_slices):
            tv.utils.save_image(template[i,:,:].to(torch.float16), gnd_dir + '/depth' + str(i) + '/gnd_img_' + str(a) + '.tif')
            ref = torch.add(ref, convfft(psf[i,:,:], add_background[i,:,:])) # Use forward projection to generate the simulated image; add all the projections together to get the final image
        
        # Save the ground truth image
        tv.utils.save_image(ref.to(torch.float16), img_dir + '/ref_img_' + str(a) + '.tif')