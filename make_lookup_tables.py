from __future__ import division

import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as sl
import scipy.integrate as spint
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
from enterprise.signals.signal_base import LogLikelihood
from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import sampler as ee_sampler
from enterprise_extensions import blocks as ee_blocks
from enterprise_extensions import deterministic

from la_forge.core import Core, load_Core
from la_forge import rednoise
from la_forge.diagnostics import plot_chains

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

pkl_path = '/home/nima/nanograv/12p5yr_bwm/channelized_12yr_v3_partim_py3.pkl'
with open(pkl_path, 'rb') as f:
    allpsrs=pickle.load(f)
psrlist = [p.name for p in allpsrs]

full_12p5_tmax_sec = max([p.toas.max() for p in allpsrs])
full_12p5_tmin_sec = min([p.toas.min() for p in allpsrs])

full_12p5_tspan_sec = full_12p5_tmax_sec - full_12p5_tmin_sec
full_12p5_tspan_mjd = full_12p5_tspan_sec/3600/24

lookup_outdir = '/home/nima/nanograv/12p5yr_bwm/lookup_tables_crn_marg/'
if not os.path.exists(lookup_outdir):
    os.mkdir(lookup_outdir)

def make_lookup_table(psr, noisefile, outdir, sign, log10_rn_amps, log10_rn_amp_spacing,
                        log10_bwm_amps, log10_bwm_amp_spacing,
                        gammas, gamma_spacing, Ts, time_spacing, log10_crn_amps, log10_crn_amp_spacing):

    if not os.path.exists(outdir + psr.name):
        os.mkdir(outdir + psr.name)

    #now we need to make a pta for this pulsar to look up likelihoods for each amplitude we calculate
    #################
    ####   PTA   ####
    #################

    # we want to introduce a fixed-index common process in addition to the usual ramp, so we'll make this by hand
    #expect Ts to be passed in units of seconds
    tmin_sec = min(Ts)
    tmax_sec = max(Ts)

    tmin_mjd = tmin_sec/3600/24
    tmax_mjd = tmax_sec/3600/24

    print("I was handed tmin: {} and tmax: {}".format(tmin_sec, tmax_sec))
    print("Expected these times to be in units of seconds")

    Tspan_sec = tmax_sec-tmin_sec
    Tspan_mjd = Tspan_sec/3600/24

    print("Tspan_sec: {}".format(Tspan_sec))
    IRN_logmin = min(log10_rn_amps)
    IRN_logmax = max(log10_rn_amps)

    CRN_logmin = min(log10_crn_amps)
    CRN_logmax = max(log10_crn_amps)

    bwm_logmin = min(log10_bwm_amps)
    bwm_logmax = max(log10_bwm_amps)



    #Intrinsic Red Noise
    s = ee_blocks.red_noise_block(psd='powerlaw', prior='log-uniform', components=30,
                                 logmin=IRN_logmin, logmax=IRN_logmax, Tspan=None)


    # Jury is still out on which Tspan to use
    # I think this needs to reflect what the Tspan will be at runtime
    # which ought to be the Tspan of the entire PTA, even if it overparameterizes
    # the individual PTA Tspan.
    #print("creating cRN with tspan = {}".format(full_12p5_tspan_sec))
    # s += ee_blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
    #                                     Tspan=full_12p5_tspan_sec, components=30, gamma_val=13./3.,
    #                                     logmin=CRN_logmin, logmax=CRN_logmax)


    # Add a ramp to the data
    print("creating ramp block with tmax, tmin: {} , {}\n and amplitudes {} , {}".format(tmin_mjd, tmax_mjd, bwm_logmin, bwm_logmax))
    s += ee_blocks.bwm_sglpsr_block(Tmin=tmin_mjd, Tmax=tmax_mjd, amp_prior='log-uniform',
                                 logmin=bwm_logmin, logmax=bwm_logmax,
                                 fixed_sign=None)

    # Common Red Noise
    # Based on results from 12p5, looks like the 13/3 process is well constrained





    s += ee_blocks.common_red_noise_block(psd='powerlaw', prior='log_uniform',
                                            Tspan=full_12p5_tspan_sec, components=5,
                                            logmin = CRN_logmin, logmax=CRN_logmax, gamma_val=13./3., name='gw')

    s += gp_signals.TimingModel()

    if 'NANOGrav' in psr.flags['pta']:
        s += ee_blocks.white_noise_block(vary=False, inc_ecorr=True)
    else:
        s += ee_blocks.white_noise_block(vary=False, inc_ecorr=False)

    models = []
    models.append(s(psr))

    pta = signal_base.PTA(models)

    print("Here are the parameters of the pta: {}".format(pta.params))
    with open(noisefile, 'rb') as nfile:
        setpars = json.load(nfile)

    pta.set_default_params(setpars)

    with open(outdir + "{}/{}_{}.txt".format(psr.name, psr.name, sign),'a+') as f:
        for t0 in Ts:
            #since t0 needs to be in mjd, we need to convert...
            t0_mjd = t0/3600/24
            for log10_strain in log10_bwm_amps:
                this_crn_amp_chart = np.zeros(len(log10_crn_amps))
                for crn_idx, log10_crn_amp in enumerate(log10_crn_amps):
                    # ====== JANKY BUT THIS IS THE ORDER OF PARAMS==============
                    # [B1855+09_red_noise_gamma:Uniform(pmin=0, pmax=7),
                    #  B1855+09_red_noise_log10_A:Uniform(pmin=-17.0, pmax=-11.0),
                    #  gw_log10_A:LinearExp(pmin=-16.0, pmax=-14.0),
                    #  ramp_log10_A:Uniform(pmin=-17.0, pmax=-11.0),
                    #  ramp_t0:Uniform(pmin=53448.0, pmax=57265.0),
                    #  sign:Uniform(pmin=-1, pmax=1)]
                    # ==========================================================

                    this_l10rn_gamma_chart = np.zeros((len(log10_rn_amps), len(gammas)))
                    for ii, log10_rn_amp in enumerate(log10_rn_amps):
                        for jj, gamma in enumerate(gammas):
                            this_l10rn_gamma_chart[ii,jj] = pta.get_lnlikelihood([gamma, log10_rn_amp, log10_crn_amp, log10_strain, t0_mjd, sign])
                    # now that we have the log likelihoods, we need to add them


                    # just to make sure that some large numbers aren't breaking things, we're going to do this in 2 steps
                    compressed_chart = np.zeros(len(log10_rn_amps))
                    for ii, l10A in enumerate(log10_rn_amps):
                        normed_margin_like = 0
                        submtx = this_l10rn_gamma_chart[ii,:]
                        maxloglike = np.amax(submtx)
                        # used simpsons rule to marginalize over gamma
                        gam_post = np.exp(submtx - maxloglike)
                        compressed_chart[ii] = np.log(spint.simpson(gam_post, x=gammas))+ maxloglike

                    # now we need to use simpsons to integrate the compressed charts
                    # since this integral is over the log of the amplitudes, we need to put in that
                    # prior-like term to make sure we integrate out over uniform amplitudes

                    # Each term in the compresed_chart is now the likelihood marginalized over rn gamma
                    compressed_max = np.amax(compressed_chart)
                    corrected_amp_post = np.exp(compressed_chart-compressed_max)

                    for ii, amp in enumerate(log10_rn_amps):
                        corrected_amp_post[ii] = 10**amp * corrected_amp_post[ii]


                    marg_over_irn_lnlike = np.log(spint.simpson(corrected_amp_post, x=log10_rn_amps)) + compressed_max
                    this_crn_amp_chart[crn_idx] = marg_over_irn_lnlike

                # Now the integral over this crn_amp_chart will give the final LogLikelihood
                crn_amp_chart_max = np.amax(this_crn_amp_chart)
                corrected_crn_likes = np.exp(this_crn_amp_chart - crn_amp_chart_max)
                for ii, amp in enumerate(log10_crn_amps):
                    corrected_crn_likes[ii] = 10**amp * corrected_crn_likes[ii]
                lnlike = np.log(spint.simpson(corrected_crn_likes, x=log10_crn_amps)) + crn_amp_chart_max
                # what a journey, time to debug
                if lnlike > 0:
                    f.write('{:.15e}\n'.format(float(lnlike)))
                else:
                    f.write('{:.14e}\n'.format(float(lnlike)))



noisefile = '/home/nima/nanograv/12p5yr_bwm/wn_noisedict.json'




for psr in allpsrs:
    psrname = psr.name
    if psrname in psrlist[36:]:
    #assert psrname[0:5] == "B1855"
    ## Build grid spacing
        iRN_amp_spacing = '-17,-11,45'
        iRN_log10_amps = np.linspace(-17, -11, 45, endpoint=True)

        bwm_amp_spacing = '-17,-11,45'
        bwm_log10_amps = np.linspace(-17, -11, 45, endpoint=True)

        gamma_spacing ='0,7,28'
        gammas = np.linspace(0, 7, 28, endpoint=True)

        crn_amp_spacing='-16,-14,20'
        crn_log10_amps = np.linspace(-16, -14, 20, endpoint=True)

        #t0min_sec = psr.toas.min()
        #t0max_sec = psr.toas.max()


        U,_ = utils.create_quantization_matrix(psr.toas, dt=7)
        eps = 9  # clip first and last N observing epochs

        t0min_sec = np.floor(max(U[:,eps] * psr.toas))
        t0max_sec = np.ceil(max(U[:,-eps] * psr.toas))

        # t0min_sec = psr.toas.min() + 180*3600*24
        # t0max_sec = psr.toas.max() - 180*3600*24

        t0min_mjd = t0min_sec/3600/24
        t0max_mjd = t0max_sec/3600/24



        # we're actually going to extend the t0s to a little before and after the pulsar's observing baseline
        # hopefully, this is enough to fix our problem with early and late times

        tspan_sec = t0max_sec - t0min_sec
        tspan_mjd = (t0max_mjd - t0min_mjd)
        tspan_months = tspan_mjd/30

        print("For PSR {} I got {} months of data".format(psr.name, tspan_months))

        epoch_steps = int(np.floor(tspan_months))

        #we're going to pass in the times as seconds, and the worker function will process the input
        Ts = np.linspace(t0min_sec, t0max_sec, num=epoch_steps, endpoint=True) # These probably need to reflect the change above in 192/193
        time_spacing = '{},{},{}'.format(t0min_mjd, t0max_mjd, epoch_steps)
        sign_spacing = '-1,1,2'


        ## Some bookkeeping

        if not os.path.exists(lookup_outdir + psr.name):
            os.mkdir(lookup_outdir + psr.name)


        with open(lookup_outdir+'{}/pars.txt'.format(psr.name), 'w+') as f:
            f.write('{};{}\n{};{}\n{};{}'.format( 'ramp_log10_A',bwm_amp_spacing, 'ramp_t0',time_spacing,'sign', sign_spacing))

        ## Let it rip! We're doing the signs in parallel to speed things up, and we'll just add them back up
        ## atoothe end.

        #psr, noisefile, outdir, sign, log10_rn_amps, log10_rn_amp_spacing,
        #                        log10_cRN_amps, log10_crn_amp_spacing, log10_bwm_amps, log10_bwm_amp_spacing,
        #                        gammas, gamma_spacing, Ts, time_spacing,

        params1=[psr, noisefile, lookup_outdir, 1, iRN_log10_amps, iRN_amp_spacing, bwm_log10_amps, bwm_amp_spacing, gammas, gamma_spacing, Ts, time_spacing, crn_log10_amps, crn_amp_spacing]
        params2=[psr, noisefile, lookup_outdir, -1, iRN_log10_amps, iRN_amp_spacing, bwm_log10_amps, bwm_amp_spacing, gammas, gamma_spacing, Ts, time_spacing, crn_log10_amps, crn_amp_spacing]

        pool = mp.Pool(2)
        pool.starmap(make_lookup_table, [params1, params2])
