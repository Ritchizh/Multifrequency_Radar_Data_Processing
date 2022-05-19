# Multifrequency_Radar_Data_Processing

Big_22.2_26.2_nf11_56x43cm_3mm_d9cm - is the radar data file;
myfile_phi.txt - contains calibration coefficients for the phase linearization;

Main_Reconstruct_and_Proc.py - the main scipt with the code to read, process and visualize data with various options;
Methods_Reconstruct_and_Proc.py - the script with all methods implementation.

-***- Options: -***-

- read the data;
- apply Gaussian window with the specified widths stdx, stdy;
- apply high-pass/low-pass/band-pass Butterworth filtering to the data k-frequency spectrum;
- reconstruct a 3-D radar image with the FFT-based back-propagation method;
- (or reconstruct a radar image at a single frequency;)
- visualize the reconstructed data in xy/zx/zy planes, applying gamma-correction and choosing a colormap,
  axes numbers can be either physical mms or arrays indexes;
- (or visualize a single radar image;) 
- move along the data slices with 'K' and 'J' keys.
- check whether the grids sampling satisfies the Nyquist and other criteria.

##--------------------------------------------------------------------------------##

For detailed information on radar image reconstruction see the classic work:

_Sheen D.M., McMakin D.L., Hall T.E., “Three-dimensional millimeterwave imaging for concealed weapon detection,” IEEE Trans. Microwave Theory Tech., vol. 49, no. 9, pp. 1581–1592, Sep. 2001._

or its recent non-destructive testing application in our paper:

_M. Chizh, A. Zhuravlev, V. Razevig and S. Ivashov, "Broadband Microwave Imaging for Foam Insulation Diagnostics," 2018 Progress in Electromagnetics Research Symposium (PIERS-Toyama), Toyama, 2018, pp. 1887-1894. DOI: 10.23919/PIERS.2018.8598093_
