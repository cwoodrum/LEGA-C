import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
import pandas as pd

import sedpy

objid = 233432
legac_uvista = fits.getdata('/Users/qt-1/Documents/LEGAC/legac_uvista.fits')
df_legac_uvista = pd.DataFrame(np.array(legac_uvista).byteswap().newbyteorder())
work_df = df_legac_uvista[df_legac_uvista['id_x'] == objid]
obj_red = float(work_df['z_spec_x'].values)
path='{}{}{}{}{}'.format('/Users/qt-1/Documents/LEGAC/spectraDR/legac_M',int(work_df['mask'].values),'_',str(objid),'_v2.0.fits')

# --------------
# Functions to load LEGA-C data
# --------------

def zp25_mags(flux_filter):
    return -2.5*np.log10(flux_filter) + 25

def mags_to_maggies(mags):
    return 10**(-0.4*mags)

def aper_to_tot(flux_aperture, flux_Ks_total, flux_Ks):
    flux_total = flux_aperture * (flux_Ks_total/flux_Ks)
    return flux_total

def phot_ready(df, phot_name, ephot_name):
    band = df[phot_name].values
    eband = df[ephot_name].values
    band_tot = aper_to_tot(band, df['Ks_tot'].values, df['Ks'].values)
    eband_tot = aper_to_tot(eband, df['eKs_tot'].values, df['eKs'].values)
    band_mags = zp25_mags(band_tot)
    eband_mags = zp25_mags(eband_tot)
    band_maggies = mags_to_maggies(band_mags)
    eband_maggies = mags_to_maggies(eband_mags)
    return band_maggies, eband_maggies

def phot_maggies(df, objid, path):
    filternames = ["UVISTA_Ks", "UVISTA_H", "UVISTA_J", "UVISTA_Y",
                   "IB427.SuprimeCam", "IB464.SuprimeCam", "IB484.SuprimeCam", "IB505.SuprimeCam",
                   "IB527.SuprimeCam", "IB574.SuprimeCam", "IB624.SuprimeCam", "IB679.SuprimeCam",
                   "IB709.SuprimeCam", "IB738.SuprimeCam", "IB767.SuprimeCam", "IB827.SuprimeCam",
                   "spitzer_irac_ch1", "spitzer_irac_ch2",
                   #"spitzer_irac_ch3", "spitzer_irac_ch4", #"spitzer_mips_24",
                   #"galex_FUV", "galex_NUV",
                   "u_megaprime_sagem",
                   "B_subaru", "V_subaru", "g_subaru", "r_subaru", "i_subaru", "z_subaru"]
                   
    Ks_maggies, eKs_maggies = phot_ready(work_df, 'Ks', 'eKs')
    H_maggies, eH_maggies = phot_ready(work_df, 'H', 'eH')
    J_maggies, eJ_maggies = phot_ready(work_df, 'J', 'eJ')
    Y_maggies, eY_maggies = phot_ready(work_df, 'Y', 'eY')
                   
    IB427_maggies, eIB427_maggies = phot_ready(work_df, 'IB427', 'eIB427')
    IB464_maggies, eIB464_maggies = phot_ready(work_df, 'IB464', 'eIB464')
    IA484_maggies, eIA484_maggies = phot_ready(work_df, 'IA484', 'eIA484')
    IB505_maggies, eIB505_maggies = phot_ready(work_df, 'IB505', 'eIB505')
    IA527_maggies, eIA527_maggies = phot_ready(work_df, 'IA527', 'eIA527')
    IB574_maggies, eIB574_maggies = phot_ready(work_df, 'IB574', 'eIB574')
    IA624_maggies, eIA624_maggies = phot_ready(work_df, 'IA624', 'eIA624')
    IA679_maggies, eIA679_maggies = phot_ready(work_df, 'IA679', 'eIA679')
    IB709_maggies, eIB709_maggies = phot_ready(work_df, 'IB709', 'eIB709')
    IA738_maggies, eIA738_maggies = phot_ready(work_df, 'IA738', 'eIA738')
    IA767_maggies, eIA767_maggies = phot_ready(work_df, 'IA767', 'eIA767')
    IB827_maggies, eIB827_maggies = phot_ready(work_df, 'IB827', 'eIB827')
                   
    ch1_maggies, ech1_maggies = phot_ready(work_df, 'ch1', 'ech1')
    ch2_maggies, ech2_maggies = phot_ready(work_df, 'ch2', 'ech2')
#    ch3_maggies, ech3_maggies = phot_ready(work_df, 'ch3', 'ech3')
#    ch4_maggies, ech4_maggies = phot_ready(work_df, 'ch4', 'ech4')
    #mips24_maggies, emips24_maggies = phot_ready(work_df, 'mips24', 'emips24')
                   
#    fuv_maggies, efuv_maggies = phot_ready(work_df, 'fuv', 'efuv')
#    nuv_maggies, enuv_maggies = phot_ready(work_df, 'nuv', 'enuv')
    u_maggies, eu_maggies = phot_ready(work_df, 'u', 'eu')
                   
    B_maggies, eB_maggies = phot_ready(work_df, 'B', 'eB')
    V_maggies, eV_maggies = phot_ready(work_df, 'V', 'eV')
    gp_maggies, egp_maggies = phot_ready(work_df, 'gp', 'egp')
    rp_maggies, erp_maggies = phot_ready(work_df, 'rp', 'erp')
    ip_maggies, eip_maggies = phot_ready(work_df, 'ip', 'eip')
    zp_maggies, ezp_maggies = phot_ready(work_df, 'zp', 'ezp')
                   
    maggies = [Ks_maggies, H_maggies, J_maggies, Y_maggies,
                IB427_maggies, IB464_maggies, IA484_maggies, IB505_maggies,
                IA527_maggies, IB574_maggies, IA624_maggies, IA679_maggies,
                IB709_maggies, IA738_maggies, IA767_maggies, IB827_maggies,
               ch1_maggies, ch2_maggies, #ch3_maggies, ch4_maggies, #mips24_maggies,
               #fuv_maggies, nuv_maggies,
               u_maggies,
                B_maggies, V_maggies, gp_maggies, rp_maggies, ip_maggies, zp_maggies]
                   
    emaggies = [eKs_maggies, eH_maggies, eJ_maggies, eY_maggies,
                eIB427_maggies, eIB464_maggies, eIA484_maggies, eIB505_maggies,
                eIA527_maggies, eIB574_maggies, eIA624_maggies, eIA679_maggies,
                eIB709_maggies, eIA738_maggies, eIA767_maggies, eIB827_maggies,
                ech1_maggies, ech2_maggies, #ech3_maggies, ech4_maggies, #emips24_maggies,
                #efuv_maggies, enuv_maggies,
                eu_maggies,
                eB_maggies, eV_maggies, egp_maggies, erp_maggies, eip_maggies, ezp_maggies]
    maggies = np.concatenate(maggies, axis=0)
    emaggies = np.concatenate(emaggies, axis=0)
    return maggies, emaggies

def prospector_wl_flux(path):
    hdul = fits.open(path)
    data = hdul[1].data
    cols = hdul[1].columns
    wl = data.field('WAVE')[0]
    wl = np.array(wl)
    flux = data.field('FLUX')[0]
    flux = np.array(flux)*1e-19
    flux_err = data.field('ERR')[0]
    flux_err = np.array(flux_err)
    zero_indices = np.where(flux == 0)[0]
    flux_nonzero = np.delete(flux, zero_indices)
    flux_err_nonzero = np.delete(flux_err, zero_indices)
    wl_nonzero = np.delete(wl, zero_indices)
    flux_good = flux_nonzero[10:-10]
    flux_err_good = flux_err_nonzero[10:-10]
    wl_good = wl_nonzero[10:-10]
    lam2_c = (wl_good*wl_good)/2.99792e18
    fl_nu = lam2_c*flux_good
    fl_jansky = fl_nu*1e23
    fl_maggies = fl_jansky/3631
    fl_err = lam2_c*flux_err_good*1e23*1e-19
    fl_err_maggies = fl_err/3631
    return wl_good, fl_maggies, fl_err_maggies

# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose':True,
              'debug':False,
              'outfile':'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol':0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers':128,
              'nburn': [16, 32, 64],
              'niter': 512,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'unif', # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
#              'objid':0,
#              'phottable': 'demo_photometry.dat',
              'luminosity_distance': None,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_duste': True,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# Model Definition
# --------------

def build_model(fixed_metallicity=None, add_duste=True,
               add_neb=True, luminosity_distance=None, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"

    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]
    
    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 5
    model_params["mass"]["init"] = 1e10

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0., maxi=4.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-3, maxi=12)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e9, maxi=1e13)
    model_params["tage"]["prior"] = priors.TopHat(mini=0., maxi=cosmo.age(obj_red).value)

    model_params["dust_type"] = {'N': 1, 'isfree': False,
        'init': 2, 'prior': None}
#    model_params["dust_index"] = {'N': 1, 'isfree': True,
#        'init': -0.7, 'prior': priors.TopHat(mini=-0.1, maxi=-.9)}
    model_params["imf_type"] = {'N': 1, 'isfree': False,
        'init': 1, 'prior': None}
    model_params["sfh"] = {'N': 1, 'isfree': False,
    'init': 1, 'prior': None}
    
    model_params['zred'] = {'N': 1, 'isfree': True, 'init': obj_red,
                "prior": priors.TopHat(mini=obj_red-0.05, maxi=obj_red+0.05)}
    
#    model_params['add_igm_absorption'] = {'N': 1, 'isfree': False, 'init': 1,
#                'units': None, 'prior_function': None, 'prior_args': None}
#    model_params['add_agb_dust_model'] = {'N': 1, 'isfree': False, 'init': 1,
#                'units': None, 'prior_function': None, 'prior_args': None}

    #order of polynomial that's fit to spectrum
    model_params['polyorder'] = {'N': 1, 'init': 10, 'isfree': False}
    # fit for normalization of spectrum
    model_params['spec_norm'] = {'N': 1, 'init': 1.0, 'isfree': True, 'prior': priors.Normal(sigma=0.2, mean=1.0), 'units': 'f_true/f_obs'}
    
    model_params['spec_jitter'] = {"N": 1, "isfree": True, "init": 1.0, "prior": priors.TopHat(mini=0., maxi=4.0)}

#    # --- Add smoothing parameters ---
#    model_params.update(TemplateLibrary["spectral_smoothing"])
#    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=150, maxi=250)


    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# Observational Data
# --------------

def build_obs(objid=objid, phottable=None,
             luminosity_distance=None, **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """

    from prospect.utils.obsutils import fix_obs

    wl, flux, flux_err = prospector_wl_flux(path)
    maggies, emaggies = phot_maggies(work_df, objid, path)

    obs = {}
    filternames = ["UVISTA_Ks", "UVISTA_H", "UVISTA_J", "UVISTA_Y",
                   "IB427.SuprimeCam", "IB464.SuprimeCam", "IB484.SuprimeCam", "IB505.SuprimeCam",
                   "IB527.SuprimeCam", "IB574.SuprimeCam", "IB624.SuprimeCam", "IB679.SuprimeCam",
                   "IB709.SuprimeCam", "IB738.SuprimeCam", "IB767.SuprimeCam", "IB827.SuprimeCam",
                   "spitzer_irac_ch1", "spitzer_irac_ch2", #"spitzer_irac_ch3", "spitzer_irac_ch4", #"spitzer_mips_24",
                   #"galex_FUV", "galex_NUV",
                   "u_megaprime_sagem",
                   "B_subaru", "V_subaru", "g_subaru", "r_subaru", "i_subaru", "z_subaru",]
    obs['filters'] = load_filters(filternames)
    obs['maggies'] = maggies
    floor_phot = 0.05*obs['maggies']
    #obs['maggies_unc'] = np.clip(emaggies, floor_phot, np.inf)
    obs['maggies_unc'] = np.sqrt(emaggies*emaggies + floor_phot*floor_phot)
    obs['wavelength'] = wl
    obs['spectrum'] = flux
    floor_spec = 0.01*obs['spectrum']
    #obs['unc'] = np.clip(flux_err, floor_spec, np.inf)
    obs['unc'] = np.sqrt(flux_err*flux_err + floor_spec*floor_spec)

    obs['objid'] = objid
    #phot_wave = [f.wave_effective for f in obs['filters']]

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)
#    plt.plot(obs['wavelength'], obs['spectrum'], 'k-')
#    plt.plot(phot_wave, obs['maggies'], 'ro')
#    plt.show()
    return obs

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    from prospect.likelihood import NoiseModel
    from prospect.likelihood.kernels import Uncorrelated
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None

#def build_noise(**extras):
#    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
#    parser.add_argument('--object_redshift', type=float, default=zred,
#                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
#    parser.add_argument('--luminosity_distance', type=float, default=1e-5,
#                        help=("Luminosity distance in Mpc. Defaults to 10pc "
#                              "(for case of absolute mags)"))
#    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
#                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=objid,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    #run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass

