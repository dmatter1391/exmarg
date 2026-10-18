"""
a nested-sampling reference for one segment: bilby with the dynesty sampler on exactly our data and our prior.

our likelihood marginalizes the sky position, the orientation, the phase, the arrival time and the distance through the
tables. bilby samples the sky position, the inclination and the polarization and marginalizes the phase, the time and the
distance on the fly. both sides see the same frequency bins, the same noise curve and the same prior:
  chirp mass, mass ratio and spin uniform in the same box,
  distance uniform in volume out to D_max = D0 / ymin (the prior of the tables),
  isotropic sky and orientation, the arrival time anywhere in the segment.

    result = exmarg.compare.run_bilby(event, data, Mc_range, q_range, chi_range, nlive=300, npool=16, label='demo')

conventions: the bilby frequency-domain strain is our data, the one-sided noise curve is 2 sig2 / T, and the segment starts at
GPS - T/2.
"""
import numpy as np

GPS = 1126259462.0


def _aligned_bbh(frequency_array, chirp_mass, mass_ratio, chi, luminosity_distance, theta_jn, phase, **kwargs):
    import bilby
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    tilt = 0.0 if chi >= 0 else np.pi
    return bilby.gw.source.lal_binary_black_hole(
        frequency_array, mass_1=m1, mass_2=m2, luminosity_distance=luminosity_distance,
        a_1=abs(chi), tilt_1=tilt, a_2=abs(chi), tilt_2=tilt, phi_12=0.0, phi_jl=0.0,
        theta_jn=theta_jn, phase=phase, **kwargs)


def _interferometers(event, data):
    import bilby
    T, fmax = event.T, event.fmax
    fs = 2 * fmax
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
    for ifo, I in zip(ifos, 'HL'):
        ifo.duration = T
        ifo.sampling_frequency = fs
        frequencies = ifo.frequency_array
        psd = np.interp(frequencies, event.f, 2 * event.sig2[I] / T, left=np.inf, right=np.inf)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=frequencies, psd_array=psd)
        strain = np.zeros(frequencies.size, complex)
        strain[np.round(event.f * T).astype(int)] = data[I]
        ifo.set_strain_data_from_frequency_domain_strain(strain, sampling_frequency=fs, duration=T, start_time=GPS - T / 2)
        ifo.minimum_frequency = event.fmin
        ifo.maximum_frequency = fmax
    return ifos


def run_bilby(event, data, Mc_range, q_range, chi_range, nlive=300, npool=1, label='segment', outdir='bilby_runs'):
    """run bilby/dynesty on one segment; returns the bilby result (posterior samples and ln evidence)"""
    import bilby
    bilby.core.utils.logger.setLevel('WARNING')
    T = event.T
    waveform = bilby.gw.WaveformGenerator(
        duration=T, sampling_frequency=2 * event.fmax, frequency_domain_source_model=_aligned_bbh,
        waveform_arguments=dict(waveform_approximant='IMRPhenomXAS', reference_frequency=event.fmin,
                                minimum_frequency=event.fmin))
    d_max = event.D0 / event.ymin
    priors = bilby.core.prior.PriorDict()
    priors['chirp_mass'] = bilby.core.prior.Uniform(*Mc_range, name='chirp_mass')
    priors['mass_ratio'] = bilby.core.prior.Uniform(*q_range, name='mass_ratio')
    priors['chi'] = bilby.core.prior.Uniform(*chi_range, name='chi')
    priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, minimum=1.0, maximum=d_max, name='luminosity_distance')
    priors['ra'] = bilby.core.prior.Uniform(0, 2 * np.pi, boundary='periodic', name='ra')
    priors['dec'] = bilby.core.prior.Cosine(name='dec')
    priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, boundary='periodic', name='psi')
    priors['phase'] = bilby.core.prior.Uniform(0, 2 * np.pi, boundary='periodic', name='phase')
    priors['geocent_time'] = bilby.core.prior.Uniform(GPS - T / 2, GPS + T / 2, name='geocent_time')
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=_interferometers(event, data), waveform_generator=waveform, priors=priors,
        distance_marginalization=True, phase_marginalization=True, time_marginalization=True)
    return bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', nlive=nlive, npool=npool,
                             dlogz=0.1, outdir=outdir, label=label, resume=False, save=True, print_method='interval-600')
