"""The bilby/dynesty runs of the paper (Figs. 20 and 21 and the timing of notebook 5), for one segment.

The segments are read from a file of data/segments/ in our frequency-domain convention: d_H, d_L (complex, one row per
segment and one value per frequency bin f), f, sig2 (the noise variance of a bin, T S(f) / 2), T, fmin and fmax.  The
segment is centered at GPS time 1e9 + T/2.  bilby gets exactly these bins, with the one-sided noise curve S(f) = 2 sig2 / T
and the strain equal to our data.

Two set-ups:
  three  chirp mass (uniform between 7 and 87 solar masses), mass ratio (uniform between 0.3 and 1) and spin
         chi_1 = chi_2 = chi_eff (uniform between -0.6 and 0.6) sampled; distance, phase and arrival time marginalized
         analytically by bilby; sky position, inclination and polarization sampled;
  two    equal masses without spin (the population analysis of the paper); chirp mass and distance sampled, which gives the
         source-frame primary mass and the redshift; phase and arrival time marginalized; sky, inclination, polarization sampled.
Common: IMRPhenomXAS, the arrival time anywhere in the segment, isotropic sky and orientation, the distance uniform in
Euclidean volume out to 82,163 Mpc, dynesty with 2000 live points and dlogz = 0.1.

    python scripts/run_bilby.py three data/segments/three_parameters.npz <row> [cores]
    python scripts/run_bilby.py two   data/segments/two_parameters.npz <row> [cores]
"""
import sys

import bilby
import numpy as np
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

NLIVE, DLOGZ, D_MAX = 2000, 0.1, 82163.0
GPS = 1.0e9                                       # start of every segment


def segment(segment_file, row):
    """one segment of a file of data/segments/ as the dictionary the functions below read"""
    z = np.load(segment_file)
    T = float(z['T'])
    sig2 = z['sig2'] if 'sig2' in z.files else z['sig2_H']
    return dict(dH=z['d_H'][row], dL=z['d_L'][row], f=z['f'], sig2=sig2, T=T, fmin=float(z['fmin']),
                fmax=float(z['fmax']), gps=GPS + T / 2)


def aligned_bbh(frequency_array, chirp_mass, mass_ratio, chi, luminosity_distance, theta_jn, phase, **kwargs):
    """IMRPhenomXAS with equal aligned spins chi on both black holes"""
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    tilt = 0.0 if chi >= 0 else np.pi
    return bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=m1, mass_2=m2, luminosity_distance=luminosity_distance,
        a_1=abs(chi), tilt_1=tilt, a_2=abs(chi), tilt_2=tilt, phi_12=0.0, phi_jl=0.0,
        theta_jn=theta_jn, phase=phase, **kwargs)


def interferometers(seg):
    """LIGO Hanford and Livingston carrying the data and the noise curve of the segment"""
    T, fmin, fmax = float(seg['T']), float(seg['fmin']), float(seg['fmax'])
    bins = np.round(seg['f'] * T).astype(int)
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo, data in zip(ifos, (seg['dH'], seg['dL'])):
        ifo.set_strain_data_from_zero_noise(2 * fmax, T, float(seg['gps']) - T / 2)
        frequencies = ifo.frequency_array
        strain = np.zeros(frequencies.size, complex)
        strain[bins] = data
        ifo.set_strain_data_from_frequency_domain_strain(strain, sampling_frequency=2 * fmax, duration=T,
                                                         start_time=float(seg['gps']) - T / 2)
        psd = np.full(frequencies.size, np.inf)
        psd[bins] = 2.0 * np.asarray(seg['sig2'], float) / T
        ok = np.isfinite(psd)
        psd = np.interp(frequencies, frequencies[ok], psd[ok], left=np.inf, right=np.inf)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=frequencies, psd_array=psd)
        ifo.minimum_frequency, ifo.maximum_frequency = fmin, fmax
    return ifos


def priors(seg, setup):
    T, gps = float(seg['T']), float(seg['gps'])
    p = bilby.core.prior.PriorDict()
    if setup == 'three':
        p['chirp_mass'] = bilby.core.prior.Uniform(7.0, 87.0, 'chirp_mass')
        p['mass_ratio'] = bilby.core.prior.Uniform(0.3, 1.0, 'mass_ratio')
        p['chi'] = bilby.core.prior.Uniform(-0.6, 0.6, 'chi')
    else:
        quarter = 0.25 ** 0.6                                                     # chirp mass / total mass for equal masses
        p['chirp_mass'] = bilby.core.prior.Uniform(10 * quarter, 500 * quarter, 'chirp_mass')
        p['mass_ratio'] = bilby.core.prior.DeltaFunction(1.0, 'mass_ratio')
        p['chi'] = bilby.core.prior.DeltaFunction(0.0, 'chi')
    p['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
    p['ra'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'ra', boundary='periodic')
    p['dec'] = bilby.core.prior.Cosine(name='dec')
    p['psi'] = bilby.core.prior.Uniform(0, np.pi, 'psi', boundary='periodic')
    p['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, minimum=90.0, maximum=D_MAX, name='luminosity_distance')
    p['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi, 'phase', boundary='periodic')
    p['geocent_time'] = bilby.core.prior.Uniform(gps - T / 2, gps + T / 2, 'geocent_time')
    return p


def run(setup, segment_file, row, cores=32):
    seg = segment(segment_file, row)
    T, fmin, fmax = float(seg['T']), float(seg['fmin']), float(seg['fmax'])
    waveform = bilby.gw.WaveformGenerator(
        duration=T, sampling_frequency=2 * fmax, frequency_domain_source_model=aligned_bbh,
        waveform_arguments=dict(waveform_approximant='IMRPhenomXAS', reference_frequency=fmin, minimum_frequency=fmin))
    prior = priors(seg, setup)
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers(seg), waveform_generator=waveform, priors=prior,
        distance_marginalization=(setup == 'three'), phase_marginalization=True, time_marginalization=True)
    extra = {} if setup == 'three' else dict(sample='rslice', bound='multi')
    return bilby.run_sampler(likelihood=likelihood, priors=prior, sampler='dynesty', nlive=NLIVE, dlogz=DLOGZ, npool=cores,
                             outdir='bilby_%s' % setup, label='%s_row%d' % (segment_file.split('/')[-1][:-4], row), **extra)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 32)
