# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:40:17 2017

@author: Margarita
"""

import numpy as np
from   scipy.constants   import c
from   numpy             import pi
import matplotlib.pyplot as plt

class RECONSTR:
        def __init__(self):
                self.Ep         = []
                self.Ec         = []
                self.f          = []
                self.nf         = []
                self.d          = []
                self.nx         = []
                self.ny         = []
                self.dx         = []
                self.dy         = []
                self.x_area     = []
                self.y_area     = []
                self.x          = []
                self.y          = []
                
                self.z          = []
                self.dz         = []
                self.nz         = []
                self.z_area     = []
                
                self.my_idx     = []
                


        def load_data(self, file_name): #loads data from file into variables                
                file        = open(file_name, 'rb')  # pass your filename
                data        = np.load(file)
                f1          = np.load(file)
                f2          = np.load(file)
                self.nf     = np.load(file)
                step_m      = np.load(file)
                self.d      = np.load(file)
                file.close()
                #----------------------------------------------
                self.f         = np.linspace(f1, f2, self.nf) 
                self.k         = 2*pi*self.f/c
                #----------------------------------------------
                # Parellel polarization:
                Ip          = data[1,:, :, :]
                Qp          = data[0,:, :, :]
                self.Ep     = Ip - 1j*Qp # nf, nx, ny
#                # Cross polarization:
#                Ic          = data[3,:, :, :]
#                Qc          = data[2,:, :, :]
#                self.Ec     = Ic - 1j*Qc # nf, nx, ny
                # Geometry:
                self.nx     = self.Ep.shape[1]
                self.ny     = self.Ep.shape[2]
                self.dx     = step_m[0]
                self.dy     = step_m[1]
                self.x_area = (self.nx-1)*self.dx
                self.y_area = (self.ny-1)*self.dy
                self.x       = np.linspace(0, self.x_area, self.nx)
                self.y       = np.linspace(0, self.y_area, self.ny)
                
        def focus_multifreq(self, nz = 80, z_area = 0.1, pol='p'): #choose Z visualization limits and the data polarization
                self.nz      = nz
                self.z_area  = z_area
                self.z       = np.linspace(0, self.z_area, self.nz)
                self.dz      = self.z[1]-self.z[0]
                #----------------------------------------------
                if pol == 'p':
                    E = self.Ep
                elif pol == 'c':
                    E = self.Ec
                else:
                    raise ValueError('Polarization argument should be p or c.')
                # Calibration & Subtract mean:
                file = open('myfile_phi.txt', 'rb') # here is your calibration data file (for phase-frequency dependence linearization)
                unpacked_f, unpacked_phi = np.loadtxt(file)
                file.close
                f_new = np.zeros(self.nf)
                for i_f in range(0,self.nf):
                    E[i_f,:,:]  -= np.mean(E[i_f,:,:]) # subtract mean
                    idx = (np.abs(unpacked_f-self.f[i_f])).argmin() #find a value nearest to the given value in the array, return its index
                    f_new[i_f] = unpacked_f[idx]
                    delta_phi = unpacked_phi[idx]
                    E[i_f,:,:] *= np.exp(1j*delta_phi)
                #----------------------------------------------
                # K-grids:
                if (self.nx & 0x01):         #x.size is odd
                    kx = np.linspace(-pi/self.dx, pi/self.dx, self.nx)
                else:                       #x.size is even
                    kx = np.linspace(-pi/self.dx, pi/self.dx*(1-2/self.nx), self.nx)
    
                if (self.ny & 0x01):         #y.size is odd
                    ky = np.linspace(-pi/self.dy, pi/self.dy, self.ny)
                else:                       #y.size is even
                    ky = np.linspace(-pi/self.dy, pi/self.dy*(1-2/self.ny), self.ny)

                if (self.z.size & 0x01):         #z.size is odd
                    kz = np.linspace(-pi/self.dz, pi/self.dz, self.nz)
                else:                       #z.size is odd
                    kz = np.linspace(-pi/self.dz, pi/self.dz*(1-2/self.nz), self.nz)
                #----------------------------------------------
                # K-spectrum:
                SE = np.zeros_like(E, dtype=np.complex)
                for fi in range(0, self.nf):
                    SE[fi,:,:] = np.fft.fftshift(np.fft.fft2(E[fi,:,:]) ) # 2d-FFT and shift
                # Interpolation:
                kzz_eq, kxx, kyy = np.meshgrid(kz, kx, ky, indexing='ij')
                SE_interp = np.zeros( (self.nz, self.nx, self.ny), dtype=np.complex )
                for xi in range(0, self.nx):
                    for yi in range(0, self.ny):
                        kz_f = 4*self.k**2-kx[xi]**2-ky[yi]**2 # non-uniform kz(f) is interpolated into uniform
                        kz_f = kz_f.clip(min = 0)
                        kz_f = np.sqrt(kz_f)
                        SE_interp[:, xi, yi] = np.interp(kz, kz_f, SE[:,xi,yi], left=0, right=0)
                # Reconstruction:
                E_rec = np.fft.ifftn(SE_interp) # 3d-IFFT
                return (E_rec)
            
            
        def focus_singlefreq(self, num_freq=0, d=0, pol='p'): # choose frequency and distance at which to reconstruct the data, also the data polarization
                if pol == 'p':
                    E = self.Ep
                elif pol == 'c':
                    E = self.Ec
                else:
                    raise ValueError('Polarization argument should be p or c.')
                #----------------------------------------------    
                if num_freq > (self.nf-1):
                    raise ValueError('Frequency number should be %s or smaller.'%(self.nf-1))
                frq_val  = self.f[num_freq]
                E        = E[num_freq,:,:]
                E       -= np.mean(E)                
                #----------------------------------------------
                # K-grids:
                if (self.nx & 0x01):         #x.size is odd
                    kx1 = np.linspace(-pi/self.dx, pi/self.dx, self.nx)
                else:                       #x.size is even
                    kx1 = np.linspace(-pi/self.dx, pi/self.dx*(1-2/self.nx), self.nx)
    
                if (self.ny & 0x01):         #y.size is odd
                    ky1 = np.linspace(-pi/self.dy, pi/self.dy, self.ny)
                else:                       #y.size is even
                    ky1 = np.linspace(-pi/self.dy, pi/self.dy*(1-2/self.ny), self.ny)

                kxx1, kyy1 = np.meshgrid(kx1, ky1, indexing='ij')
                k1         = (2*pi*frq_val)/c
                #----------------------------------------------             
                SE1       = np.fft.fftshift(np.fft.fft2(E)) # 2d-FFT
                TransMx1  = np.exp(np.lib.scimath.sqrt(4*k1**2 - (kxx1)**2 - (kyy1)**2)*d*1j)
                FE1       = SE1*TransMx1
                E_rec1    = np.fft.ifft2(FE1) # 2d-IFFT
                return(E_rec1)


        def plot_E_singlefreq(self, pic_num, E, cm='gray'): # choose a colormap!
                cm = 'plt.cm.' + cm
                plt.figure(pic_num)
                pic_num +=1
                plt.imshow(abs(E), extent=[0,self.y_area*10**3, 0,self.x_area*10**3], cmap=eval(cm)) #z,x,y #extent=[0,y_area, 0,x_area]
                plt.title('Exy at 1 freq') 
                plt.xlabel('x, mm')
                plt.ylabel('y, mm')
                return (pic_num) 
                        
            
        def plot_E_multifreq(self, pic_num, E, view, num_x=0, num_y=0, d=0, cm='gray'): # choose a view and a colormap!
                cm = 'plt.cm.' + cm
                plt.figure(pic_num)
                pic_num +=1
                if view == 'xy':
                    z_indx = (np.abs(self.z-d)).argmin()
                    plt.imshow(abs(E[z_indx,:,:]), extent=[0,self.y_area*10**3, 0,self.x_area*10**3], cmap=eval(cm)) #z,x,y #extent=[0,y_area, 0,x_area]
                    plt.title('Exy multifreq') 
                    plt.xlabel('x, mm')
                    plt.ylabel('y, mm')
                elif view == 'zy':
                    plt.imshow( abs(E[:,num_x,:]), extent=[0,self.y_area*10**3, self.z_area*10**3,0], cmap=eval(cm)) #z,x,y  #extent=[0,y_area, z_area,0]
                    plt.title('Ezy multifreq') 
                    plt.xlabel('y, mm')
                    plt.ylabel('z, mm')
                elif view == 'zx':
                    plt.imshow( abs(E[:,:,num_y]), extent=[0,self.x_area*10**3, self.z_area*10**3,0], cmap=eval(cm)) #z,x,y  #extent=[0,x_area, z_area,0]
                    plt.title('Ezx multifreq') 
                    plt.xlabel('x, mm')
                    plt.ylabel('z, mm')
                else:
                    raise ValueError('view argument should be xy, zy or zx.')
                return (pic_num) 

                    
        def test_sampling(self): # check whether your geometry is Nyquist&other-criteria satisfying
            dxy_max = c/(4*self.f[-1])     
            if self.dx > dxy_max:
                print("\n- Warning: aliasing, dx = %s is too large, should be smaller than %s!" %(round(self.dx,4), round(dxy_max,4)))
            else:
                print("\n+ Note: dx sampling interval is OK!")
            if self.dy > dxy_max:
                print("\n- Warning: aliasing, dy = %s is too large, should be smaller than %s!" %(round(self.dy,4), round(dxy_max,4)))
            else:
                print("\n+ Note: dy sampling interval is OK!")
            #----------------------------------------------    
            from math import ceil    
            z_max  = c/(4*(self.f[1]-self.f[0]))
            nf_min = ceil(4*self.z_area*(self.f[-1]-self.f[0])/c)
            if self.z_area > z_max:
                print("\n- Warning: target too far, make z_area smaller than %s or nf larger than %s" %(round(self.z_max,2), nf_min))
            else:
                print("\n+ Note: z_max is OK!")
            #----------------------------------------------
            nz_min = int(self.nf*2.5)   
            if self.nz < nz_min:
                print("\n- Warning: bad interpolation, make nz larger than %s" %nz_min)
            else:
                print("\n+ Note: interpolation is OK!")     
            phi_SARx = np.arctan(self.x_area/(2*self.d))
            delta_x = c/(4*self.f[0]*np.sin(phi_SARx))
            if 2.5*self.dx < delta_x:
                print("\n- Warning: x cross range resolution is %s larger than double step 2.5*dx = %s. Make x_area larger or target depth smaller." %(round(delta_x, 4), round(2.5*self.dx, 4)))
            else:
                print("\n+ Note: x_area aperture is OK!")
            #---------------------------------------------- 
            phi_SARy = np.arctan(self.y_area/(2*self.d))
            delta_y = c/(4*self.f[0]*np.sin(phi_SARy))
            if 2.5*self.dy < delta_y:
                print("\n- Warning: y cross range resolution is %s larger than double step 2.5*dy = %s. Make y_area larger or target depth smaller." %(round(delta_y, 4), round(2.5*self.dy, 4)))
            else:
                print("\n+ Note: y_area aperture is OK!")
                
#=============================================================
# VIEW SLICES:
    
        def remove_keymap_conflicts(self, new_keys_set):
            for prop in plt.rcParams:
                if prop.startswith('keymap.'):
                    keys = plt.rcParams[prop]
                    remove_list = set(keys) & new_keys_set
                    for key in remove_list:
                        keys.remove(key)
                
        def multi_slice_viewer(self, pic_num, volume, view, disp='phys', cm='gray'): # choose a view and a colormap!
            self.remove_keymap_conflicts({'j', 'k'})
            cm = 'plt.cm.' + cm
            fig = plt.figure(pic_num)
            ax  = fig.add_subplot(111)
            if view == 'xy':
                self.my_idx = 0
                z_indx      = (np.abs(self.z-self.d)).argmin()
                ax.volume   = volume
                ax.index    = z_indx
                if disp == 'phys': # display physical coordinates
                    ax.imshow(abs(volume[ax.index,:,:]), extent=[0,self.y_area*10**3, 0,self.x_area*10**3], cmap=eval(cm)) #z,x,y #extent=[0,y_area, 0,x_area]
                    plt.xlabel('x, mm')
                    plt.ylabel('y, mm')
                elif disp == 'num': # display arrays indexes
                    ax.imshow(abs(volume[ax.index,:,:]), cmap=eval(cm)) #z,x,y
                    plt.xlabel('num_x')
                    plt.ylabel('num_y')
#                else:
#                    raise ValueError('ax argument should be phys or num.')
                plt.title('Exy multifreq')
            elif view == 'zy':
                self.my_idx = 1
                ax.volume   = volume
                ax.index    = self.nx//2
                if disp == 'phys': # display physical coordinates
                    ax.imshow( abs(volume[:,ax.index,:]), extent=[0,self.y_area*10**3, self.z_area*10**3,0], cmap=eval(cm)) #z,x,y  #extent=[0,y_area, z_area,0]
                    plt.xlabel('y, mm')
                    plt.ylabel('z, mm')
                elif disp == 'num': # display arrays indexes
                    ax.imshow( abs(volume[:,ax.index,:]), cmap=eval(cm)) #z,x,y
                    plt.xlabel('num_y')
                    plt.ylabel('num_z')
                else:
                    raise ValueError('disp argument should be phys or num.')
                plt.title('Ezy multifreq')  
            elif view == 'zx':
                self.my_idx = 2
                ax.volume   = volume
                ax.index    = self.ny//2
                if disp == 'phys': # display physical coordinates
                    ax.imshow( abs(volume[:,:,ax.index]), extent=[0,self.x_area*10**3, self.z_area*10**3,0], cmap=eval(cm)) #z,x,y  #extent=[0,x_area, z_area,0]
                    plt.xlabel('x, mm')
                    plt.ylabel('z, mm')
                elif disp == 'num': # display arrays indexes
                    ax.imshow( abs(volume[:,:,ax.index]), cmap=eval(cm)) #z,x,y  
                    plt.xlabel('num_x')
                    plt.ylabel('num_z')
                else:
                    raise ValueError('disp argument should be phys or num.')
                plt.title('Ezx multifreq')
            else:
                raise ValueError('view argument should be xy, zy or zx.')
            fig.canvas.mpl_connect('key_press_event', self.process_key)
            pic_num +=1
            return (pic_num)
        
        def process_key(self, event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == 'j':         # Events are K and J keys pressed!
                self.previous_slice(ax)
            elif event.key == 'k':
                self.next_slice(ax)
            fig.canvas.draw()
        
        def previous_slice(self, ax):
            volume = ax.volume
            ax.index = (ax.index - 1) % volume.shape[self.my_idx]  # wrap around using %
            if self.my_idx == 0:
                ax.images[0].set_array(volume[ax.index,:,:])
                str0 = 'z_index = '+str(ax.index)+', z_value = '+str(round(self.z[ax.index]*10**3,1))+' mm'
                print(str0)
            elif self.my_idx == 1:
                ax.images[0].set_array(volume[:,ax.index,:])
                str1 = 'x_index = '+str(ax.index)+', x_value = '+str(round(self.x[ax.index]*10**3,1))+' mm'
                print(str1)
            elif self.my_idx == 2:
                ax.images[0].set_array(volume[:,:,ax.index])
                str2 = 'y_index = '+str(ax.index)+', y_value = '+str(round(self.y[ax.index]*10**3,1))+' mm'
                print(str2)
        
        def next_slice(self, ax):
            volume = ax.volume
            ax.index = (ax.index + 1) % volume.shape[self.my_idx]
            if self.my_idx == 0:
                ax.images[0].set_array(volume[ax.index,:,:])
                str0 = 'z_index = '+str(ax.index)+', z_value = '+str(round(self.z[ax.index]*10**3,1))+' mm'
                print(str0)
            elif self.my_idx == 1:
                ax.images[0].set_array(volume[:,ax.index,:])
                str1 = 'x_index = '+str(ax.index)+', x_value = '+str(round(self.x[ax.index]*10**3,1))+' mm'
                print(str1)
            elif self.my_idx == 2:
                ax.images[0].set_array(volume[:,:,ax.index])
                str2 = 'y_index = '+str(ax.index)+', y_value = '+str(round(self.y[ax.index]*10**3,1))+' mm'
                print(str2)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                