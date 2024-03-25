# Improt dependencies
import torch
import gc
import os
from fft_conv import FFT_Conv2d

# Define the deconvolution function
def rl_deconv(frames, PSF, PSF_bp, fov_radius, mask_width, device, iterations = 10, path = r"./", cpu_option = 'cpu', temp_storage = False, save_every = (False, 0), auto_stop = False, background_subtract = False):
    # Take the whole lenslet image & the whole PSFs and apply the Richardson-Lucy algorithm to it
    # Frames should be in the format of [time, x, y]
    # PSF should be in the format of [depths, x, y]
    # Center_lenslet_pixel is the center of the lenslet in pixels (x,y)
    # Output should be in the format of [time, depth, x, y]
    # cpu_option is the device that does the division and the multiplication.
    # temp_storage dictates how the temporary storage is handled. If it is set to True, then the temporary storage is saved to the disk. If it is set to False, then the temporary storage is saved to the RAM. Overwrite this by setting the "cpu_option" to 'cuda' or whatever device you want to use.
    # The problem is that the memory is not being freed up after each iteration. This is a problem with the way that the memory is being handled.
    
    diagnostics = False # Saves EVERY STEP to the disk. Use this to debug the code. This will take up a lot of space and be very slow.
    
    if diagnostics == True:
        print("Diagnostics mode is on. This will save EVERY step to the disk. This will take up a lot of space and be very slow.")
    
    if diagnostics == True and iterations != 1:
        print("Warning: You will overwrite your diagnostics files with each iterations if iterations != 1.")
    
    with torch.no_grad():
        
        if type(frames) != torch.Tensor:
            frames = torch.tensor(frames)
            gc.collect()
        
        if frames.dtype != torch.float32:
            frames = frames.to(torch.float32)
            gc.collect()
        
        def box_mask(img, forward_projection, square_length, tolerance):
                box_kernel = torch.zeros(img.shape).to(cpu_option)
                box_kernel[int(img.shape[0]/2)-int(square_length/2):int(img.shape[0]/2)+int(square_length/2), int(img.shape[1]/2)-int(square_length/2):int(img.shape[1]/2)+int(square_length/2)] = 1
                peaks = forward_projection[(len(forward_projection[:,0,0])//2),:,:].to(cpu_option)
                mask_box = convfft(peaks, box_kernel)
                mask_box[mask_box <= tolerance] = 0
                mask_box[mask_box > tolerance] = 1
                mask_box = torch.mul(mask_box, img)
                
                return mask_box

        def center_pix(forward_projection):
            peaks = forward_projection[int(len(forward_projection[:,0,0])//2),:,:]
            center = (peaks == torch.max(peaks)).nonzero()[0].tolist()
            return center
        
        center_lenslet_pixel = center_pix(PSF)
        
        for k in range(len(frames[:,0,0])):
            # Each time step
            original_image = box_mask(frames[k,:,:].detach().to(cpu_option), PSF, mask_width, 0.1) # Format: (x,y)
            
            if background_subtract == True:
                original_image = original_image - torch.mean(original_image, dim=0, keepdim=True)
            
            original_image_max_intensity = int(torch.max(original_image)) # Need this so that we can ignore numerical instability from not completely covering the lenslet.
            projected_volume = torch.ones([len(PSF[:,0,0]), len(frames[0,:,0]), len(frames[0,0,:])], dtype = torch.float32).detach().to(cpu_option) # Format: (depth, x,y)
            
            def circ(x, y, r):
                return ((x - center_lenslet_pixel[0]) **2 + (y - center_lenslet_pixel[1]) **2 < r**2).to(int)
            
            x_1 = torch.linspace(0, int(original_image.shape[0]), original_image.shape[0]).to(cpu_option)
            y_1 = torch.linspace(0, int(original_image.shape[1]), original_image.shape[1]).to(cpu_option)
            x, y = torch.meshgrid(x_1, y_1, indexing = 'xy')
            
            projected_volume = torch.mul(projected_volume, circ(x, y, fov_radius).unsqueeze(0))
            
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            print("Finished creating the projected volume and image for frame " + str(k+1) + " out of " + str(len(frames[:,0,0])) + ", now processing the iterations.")
            
            for j in range(iterations):
                print("Now processing iteration " + str(j+1) + " out of " + str(iterations) + " for frame " + str(k+1) + " out of " + str(len(frames[:,0,0])))
                # Schema:
                # Calculate the projected image (Sum of (projected_volume * PSF) = projected_image
                # Divide the original image by the projected image (original_image / projected_image) = ratio
                # Convolve the ratio with the backward projector (ratio * PSF_bp) = result
                # Multiply the projected volume by the result of the convolution (result * projected_volume)
                # Repeat until the iterations are done

                # Denominator results in a projected image
                projected_image = torch.empty(frames[k,:,:].shape).to(cpu_option)
                
                for l in range(len(PSF[:,0,0])):
                    if temp_storage == True:
                        temp_vol = torch.load(path + 'tempvol' + '_' + str(l) + '.pt').to(device)
                        projected_image += torch.nan_to_num(convfft(temp_vol.detach().to(device), PSF[l,:,:].detach().to(device)), nan=0, posinf=0, neginf =0).to(cpu_option)
                        del temp_vol
                        gc.collect()
                    
                    if temp_storage == False:
                        projected_image += torch.nan_to_num(convfft(projected_volume[l,:,:].to(device), PSF[l,:,:].detach().to(device)), nan=0, posinf=0, neginf=0).to(cpu_option)
                        
                    if diagnostics == True:
                        torch.save(projected_image.detach().to(cpu_option), path + 'projected_image1' + '_' + str(l) + '.pt')
                        
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                projected_image[projected_image > (10*original_image_max_intensity)] = 10*original_image_max_intensity # Don't let a pixel be more than 10 times the maximum intensity of the original image; instead set it to the maximum intensity of the original image.
                #projected_image = original_image_energy * torch.div(projected_image, int(torch.sum(projected_image)))
                
                if auto_stop == True:
                # New: We also want to take the histogram and see if it's worth keep going with the iterations.
                    if j == 0:
                        # Let the first and second iteration go through
                        pass
                    
                    elif j == 1:
                        g_p = torch.histc(projected_image, bins = 1000, min = 0, max = 1) # Maximum might have to be changed to the maximum value of the projected image.
                    
                    else:
                        g_c = torch.histc(projected_image, bins = 1000, min = 0, max = 1) # Maximum might have to be changed to the maximum value of the projected image.
                        difference = torch.abs(g_c-g_p) # This is the difference between the two histograms/iterations.
                        
                        if torch.sum(torch.gradient(difference)[0]) < 0:
                            print("The intensity histogram has converged. Stopping the iterations as further iterations will may yield noisy results.")
                            print("Saving iteration number " + str(j+1) +  " results and moving on to the next frame...")
                            
                            if temp_storage == False:
                                torch.save(projected_volume.to(torch.float32), path + str(k) + '.pt')
                    
                            if temp_storage == True:
                                for m in range(len(PSF[:,0,0])):
                                    os.rename(path + 'tempvol' + '_' + str(m) + '.pt', path + str(k) + '_' + str(m) + '.pt')
                            
                            break
                        
                        g_p = g_c
                        
                        del g_c
                        gc.collect()
                        
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                
                image_ratio = torch.nan_to_num(torch.div(original_image, projected_image), nan = 0, posinf=0 ,neginf=0).to(cpu_option) # Numerical instability starts here...
                
                if diagnostics == True:
                    torch.save(image_ratio.detach().to(cpu_option), path + 'image_ratio' + '_' + str(k) + '.pt')
                
                # Convolve the ratio with the backward projector
                
                factor = torch.empty([len(PSF[:,0,0]),len(PSF[0,:,0]), len(PSF[0,0,:])], dtype = torch.float32).to(cpu_option) # Format: (depth,x,y)\
                
                for l in range(len(PSF_bp[:,0,0])):
                    #print("Now creating multiplying factor " + str(i+1) + " out of " + str(len(PSF_bp[:,0,0])) + " for iteration " + str(j+1) + " out of " + str(iterations) + " for frame " + str(k+1) + " out of " + str(len(frames[:,0,0])))
                    factor_results = (1e5)*torch.abs(torch.nan_to_num(convfft((1e-5)*image_ratio.to(device), PSF_bp[l,:,:].detach().to(device)).to(device), nan =0, posinf=0, neginf =0)).to(cpu_option).to(cpu_option)
                    
                    if temp_storage == True:
                        torch.save(factor_results, path + 'factor' + '_' + str(l) + '.pt')
                        
                    elif temp_storage == False:
                        factor[l,:,:] = factor_results.to(cpu_option)
                        
                    if diagnostics == True:
                        torch.save(factor[l,:,:].detach(), path + 'factor_results' + '_' + str(l) + '.pt')
                    
                    del factor_results
                    gc.collect()
                    
                del image_ratio # We don't need this anymore
                gc.collect()
                
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                #factor[factor > 1e6] = 1e6 # Numerical instability starts here... (Clamp the maximum error propagation to 1e6)
                
                # We have the new factor, now multiply it by the original volume
                if temp_storage == True:
                    for m in range(len(PSF[:,0,0])):
                        temp_vol = torch.load(path + 'tempvol' + '_' + str(m) + '.pt').to(cpu_option)
                        temp_factor = torch.load(path + 'factor' + '_' + str(m) + '.pt').to(cpu_option)
                        
                        if diagnostics == True:
                            torch.save(torch.mul(temp_vol, temp_factor), path +'tempvol'+ '_' + str(m) + '.pt')
                        
                        del temp_vol, temp_factor
                        gc.collect()
                        
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                
                if temp_storage == False:
                    projected_volume[:,:,:] = len(PSF[:,0,0]) * torch.mul(projected_volume[:,:,:].to(cpu_option), factor[:,:,:].to(cpu_option)).to(cpu_option)
                    
                    if diagnostics == True:
                        torch.save(projected_volume[0,:,:].detach().to(cpu_option), path + 'projected_volume' + '_' + str(k) + '.pt')
                
                if save_every[0] == True:
                    if j % save_every[1] == 0:
                        torch.save(projected_volume.to(torch.float32), path + str(k) + 'itr' + str(j) + '.pt')
            
            print("Finished processing frame " + str(k+1) + " out of " + str(len(frames[:,0,0])) + ", now saving the results.")
            
            if temp_storage == False:
                torch.save(projected_volume[:, center_lenslet_pixel[1] - fov_radius: center_lenslet_pixel[1] + fov_radius, center_lenslet_pixel[0] - fov_radius: center_lenslet_pixel[0] + fov_radius].to(torch.float32), path + str(k) + '.pt')
                
            if temp_storage == True:
                for m in range(len(PSF[:,0,0])):
                    os.rename(path + 'tempvol' + '_' + str(m) + '.pt', path + str(k) + '_' + str(m) + '.pt')
        
        del projected_volume, projected_image, original_image, factor
        gc.collect()
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    return None