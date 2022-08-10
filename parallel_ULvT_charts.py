from __future__ import division

import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as sl
import healpy as hp
import multiprocessing as mp
import math

os.environ["TEMPO2"]='/home/nima/.local/share/tempo2/'

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise import constants as const


from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import sampler as ee_sampler
from enterprise_extensions import blocks as ee_blocks
from enterprise_extensions import deterministic

from la_forge.core import Core, load_Core
from la_forge import rednoise
from la_forge.diagnostics import plot_chains

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

nano12_Ts = np.linspace(53216.13175403865+180, 57933.45642396011-180, 157)

def calculate_angular_response(gwphi, gwtheta, gwpol, skyposition):

    # calculate glitch amplitude for pulsar
    # This is just copy-pasted from ramp_delay in enterprise.signals.utils
    apc = utils.create_gw_antenna_pattern(skyposition, gwtheta, gwphi)

    # grab fplus, fcross
    fp, fc = apc[0], apc[1]
    # combined polarization
    angular_response = np.cos(2 *gwpol) * fp + np.sin(2 * gwpol) * fc

    return angular_response

def find_closest(mytarget, mylist, is_epoch):
        """
        Just a little utility function that spits out the index of the closest value to a target in some list
        """
        if mytarget < mylist[0] and is_epoch:
            return -1
        elif mytarget < mylist[0] and not is_epoch:
            return 0
        else:
            closest = 0
            diff = np.abs(mytarget - mylist[0])
            for i, val in enumerate(mylist):
                if np.abs(mytarget - val) < diff:
                    diff = np.abs(mytarget - val)
                    closest = i
            return closest

def read_likelihood(lookup_file, line_no):
    """
    This function just looks up the likelihood from a likelihood lookup file
    We expect that the lookup_files are large (10^8 lines), so we have to do some gymnastics to do this in less than O(numlines)
    """
    with open(lookup_file, 'rb') as f:
        #find the number of bytes in one line
        bytes_per_line = 0
        while f.read(1) != b'\n':
            bytes_per_line += 1
    with open(lookup_file, 'r') as f:
        f.seek((bytes_per_line+1) * (line_no-1), 0)
        output = f.read(bytes_per_line) #decode the binary characters


    loglike = float(output)
    return loglike


def lookup_likelihood(lookupdir, target_dict, psrname):
    # This function will lookup the likelihood of a particular
    # pulsar's parameters.
    # The dictionary should have a gw_log10_A, ramp_log10_A, ramp_t0 and sign

    #print(target_dict)
    #print("looking for target dict: {}".format(target_dict))

    ###==============================================================
    ### CURRRENT VERSION USING EVENLY SPACED GRIDDING
    ###==============================================================
    pfile_dict={}
    parfile = lookupdir + '{}/pars.txt'.format(psrname)
    with open(parfile, 'r') as pfile:
        pfile_lines = pfile.readlines()
    lens = []
    for line in pfile_lines:
        key = line.split(";")[0]
        val = line.split(";")[1]

        start,stop,num = val.split(",")
        start=float(start)
        stop=float(stop)
        num=int(num)
        lens.append(num)
        arr_val = np.linspace(start, stop, num, endpoint=True)

        pfile_dict[key] = arr_val
    ###==============================================================

    ### IF I WANT TO MAKE THIS MORE ROBUST, SHOULD REPLACE THE ABOVE with
    ### code that directly reads in the gridding as arrays from a parfile (need to change 'make_lookup_table.py')
    ### Other than that, I think the index reading won't need to actually change.

    #print("searching through {}".format(pfile_dict))
    # Now I have an array attached to each parameter name
    # I need to find the matching indices of the par_dict

    idxs = []
    for curr_key in pfile_dict:

        #print("Looking for {} in array: {}".format(target_dict[curr_key], pfile_dict[curr_key]))

        target_val = target_dict[curr_key]
        if curr_key == 'ramp_t0':
            is_epoch=True
        else:
            is_epoch = False
        idx_closest = find_closest(target_val, pfile_dict[curr_key], is_epoch=is_epoch)
        #print("Found {} as idx {} in {}".format(target_val, idx_closest, pfile_dict[curr_key]))
        idxs.append(idx_closest)

    # now idxs contains the closest indices.
    # let's check to make sure none of these are outside of priors first
    # if they are, short circuit return 0
    # print(idxs)
    for ii, idx in enumerate(idxs):
        if idx < 0:
            return 0

    #If not short circuited, we need to calculate the relevant line in the lookup_table
    #and return the log likelihood
    line_no = 1
    #print("finding line number of idxs: {}".format(idxs))
    for ii, idx in enumerate(idxs):
        line_change = idx
        for jj in range(0, ii):
            line_change *= lens[jj]

        line_no += line_change

    if line_no < 0:
        print("Encountered negative line number to seek")
    lookup_file = lookupdir + '{}/{}_lookup.txt'.format(psrname,psrname)
    log_like = read_likelihood(lookup_file, line_no)
    return log_like


def calculate_likelihood(bwm_theta, bwm_phi, bwm_log10_A, bwm_pol,
                         bwm_epoch,
                         psrs,
                         lookupdir):

    #This function will calculate the likelihood of a burst across all pulsars



    total_loglikelihood = 0
    for psr in psrs:
        # i think we should probably check to see if the epoch is too early here, and ignore
        # pulsars for which the epoch isn't anywhere in the data

        #if psr.toas[0]/3600/24 > bwm_epoch:
        #    this_loglikelihood = 0
        #else:
        psrpos = psr.pos
        ang_resp = calculate_angular_response(bwm_phi, bwm_theta, bwm_pol, psrpos)

        target_dict = {}

        ramp_amp = (10**bwm_log10_A) * ang_resp

        sign = np.sign(ramp_amp)
        ramp_log10_A = np.log10(np.abs(ramp_amp))

        target_dict['ramp_log10_A'] = ramp_log10_A
        target_dict['sign'] = sign
        target_dict['ramp_t0'] = bwm_epoch


        this_lookupdir = lookupdir

        #print("Looking up for pulsar {}".format(psr.name))
        this_loglikelihood = lookup_likelihood(this_lookupdir,target_dict, psr.name)

        total_loglikelihood += this_loglikelihood

    return total_loglikelihood


    # Going to relabel some things to parallelize this... it takes too long
# 57697.590190464034
# 53451.99798753473
def likelihood_chart_wrapper(ipix, nside=2, Ts=nano12_Ts):
    import time
    start_time = time.time()
    outdir =  '/home/nima/nanograv/12p5yr_bwm/ULvT_charts_crnmarg/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)



    nano12_pklfile = '/home/nima/nanograv/12p5yr_bwm/channelized_12yr_v3_partim_py3.pkl'

    with open(nano12_pklfile, 'rb') as f:
        psrs=pickle.load(f)


    bwm_pols = np.arange(0, np.pi, np.pi/8)
    bwm_log10_As = np.linspace(-17, -11, 45, endpoint=True)



    bwm_theta, bwm_phi = hp.pix2ang(nside, ipix)

    print("beginning the loop through everything")
    for ii, bwm_pol in enumerate(bwm_pols):

        loglike_chart_outfile = outdir + '{}_{}.txt'.format(ipix, ii)
        print("Will write results in {}".format(loglike_chart_outfile))
        loglike_chart = np.zeros((len(bwm_log10_As), len(Ts)))

        for i, bwm_log10_A in enumerate(bwm_log10_As):
            for j, burst_epoch in enumerate(Ts):

                loglike_chart [i,j] = calculate_likelihood(bwm_theta, bwm_phi, bwm_log10_A,
                                     bwm_pol, burst_epoch,
                                     psrs,
                                     lookupdir = '/home/nima/nanograv/12p5yr_bwm/lookup_tables_crn_marg/')
        np.savetxt(loglike_chart_outfile, loglike_chart)
        #print("completed lookup for amplitude and epoch [{}  {}], loglike = {}".format(i,j, this_t0_l10A_loglike))

    timediff = time.time()-start_time
    print("finished charts for {} in {} seconds".format(ipix, timediff))




nside = 2
npix = hp.nside2npix(nside)
print(npix)
skymap = np.arange(npix)

mypool = mp.Pool(processes=12)

mypool.map(likelihood_chart_wrapper,skymap)
