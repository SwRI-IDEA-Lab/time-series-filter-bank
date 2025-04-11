import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

import os,sys

_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_MODEL_DIR)
sys.path.append(_SRC_DIR)

import tsFB.utils.CR_dates as crdt

# %%
hdu_list = fits.open('/home/jkobayashi/gh_repos/time-series-filter-bank/data/FITS/IDSEAR_AIAsyn/aia193_synmap_cr2098.fits')
hdu_list.info()
# %%
image_data = hdu_list[0].data
print(image_data.shape)
image_data
# %%
hdu_list.close()
# %% test plot
plt.imshow(image_data[::-1,:],cmap='gray')
# %%
plt.imshow(image_data[::-1,::-1],cmap='gray')

# %%
CR_dates = crdt.create_CR_date_dictionary('/home/jkobayashi/gh_repos/time-series-filter-bank/data/CR_Table.rdb.txt')
cr_start,cr_end = crdt.get_start_end_dates(CR_dates=CR_dates,
                                     carr_rot_num='2098')