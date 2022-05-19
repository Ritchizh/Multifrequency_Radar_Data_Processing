# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:41:24 2017

@author: Margarita
"""

#import numpy as np
import matplotlib.pyplot as plt

## On Windows may need to add project folder to path to be able to import Python script as a module:
# import sys
# sys.path.append('C:/Users/Margarita/Desktop/Multifrequency_Radar_Data_Processing-master/')

import Methods_Reconstruct_and_Proc #import class with the processing methods
pr = Methods_Reconstruct_and_Proc.RECONSTR()

#==================================================================
# MAIN:
E = pr.load_data('Big_22.2_26.2_nf11_56x43cm_3mm_d9cm.npy') #filename
#E = pr.gauss_wind(E, stdx = 200, stdy = 1000) # stdx, stdy - window width

# Options: butter_filter(E, mode, n=1, D_h=1, D_l=1000): - n is filter order; choose D_h and D_l for a high/low/band-pass
E = pr.butter_filter(E, mode='B', D_h=1, D_l=1000)
            
# Options: focus_multifreq(nz = 80, z_area = 0.1, pol='p' or 'c') - z can be any area you want to vizualize
E_rec = pr.focus_multifreq(E, nz=110, z_area=0.18) # at all frequencies

#==================================================================

#PLOTS:   
pic_num = 1

# Options:    
# show one slice - 
#pic_num = pr.plot_E_multifreq(pic_num, E_rec, view='zy', num_x=40, clrmap='gray')
#pic_num = pr.plot_E_multifreq(pic_num, E_rec, view='xy', d=pr.d,   clrmap='gray')
#pic_num = pr.plot_E_multifreq(pic_num, E_rec, view='zx', num_y=42, clrmap='gray')


#==================================================================
# Options:
# multi_slice_viewer(pic_num, my_data, view = 'xy','zx' or 'zy',  disp='phys' or 'num',gamma<1 expands low int, cm='gray' or 'other')
pic_num = pr.multi_slice_viewer(pic_num, abs(E_rec), view='xy', disp='num', gamma= 1.2, cm='gray')
#==================================================================
plt.show()
