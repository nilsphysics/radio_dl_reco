import numpy as np
from NuRadioReco.utilities import units
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioMC.EvtGen import generator
from NuRadioMC.simulation import simulation
import os
import time
import secrets
import argparse
from NuRadioMC.utilities import runner
import argparse
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
import logging
import copy
import yaml

root_seed = secrets.randbits(128)

# 4 channel PA: 2x sampling, fft upsampling, 16 ns window, 11 beams
# Linear fit: -0.2519 x + 0.727
# Quadratic fit: -0.0002581 x x - 0.2393 x + 0.5845
# Noise rate -> threshold
#
# 10000.000 Hz -> 22.74
# 100.000 Hz -> 30.68
# 1.000 Hz -> 38.62
# 0.001 Hz -> 50.53

# 8 Channel PA: 4x sampling, fft upsampling, 16 ns window, 21 beams
# Linear fit: -0.134 x + 1.295
# Quadratic fit: -0.0007308 x x - 0.06398 x - 0.2857
# Noise rate -> threshold
#
# 10000.000 Hz -> 46.98
# 100.000 Hz -> 61.90
# 1.000 Hz -> 76.83
# 0.001 Hz -> 99.22

thresholds_pa = {}
thresholds_pa['8ch'] = {}
thresholds_pa['8ch']['100Hz'] = 61.90
thresholds_pa['8ch']['1Hz'] = 76.83
thresholds_pa['8ch']['1mHz'] = 99.22

thresholds_pa['4ch'] = {}
thresholds_pa['4ch']['100Hz'] = 30.68
thresholds_pa['4ch']['1Hz'] = 38.62
thresholds_pa['4ch']['1mHz'] = 50.53 

# initialize detector sim modules
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator(log_level=logging.WARNING)
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster(log_level=logging.WARNING)
triggerTimeAdjuster.begin(pre_trigger_time=200*units.ns)

# DEEP part
# assuming that PA consists out of 4 antennas (channel 0-7)
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))

#shallow_channels = [0, 1, 2, 3]
#PA_8ch_channels = [4, 5, 6, 7, 8, 9, 10, 11]
#PA_4ch_channels = [8, 9, 10, 11]
PA_4ch_channels = [0, 1, 2, 3]
reco_channels = range(4, 16)

passband_low = {}
passband_low_trigger = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}
for channel_id in PA_4ch_channels:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = 'cheby1'
    order_low[channel_id] = 7
    order_high[channel_id] = 4
    
for channel_id in reco_channels:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = 'cheby1'
    order_low[channel_id] = 7
    order_high[channel_id] = 4

thresholds = {
    '2/4_100Hz': 3.9498194908011524,
    '2/4_10mHz': 4.919151494949084,
    'fhigh': 0.15,
    'flow': 0.08}

"""
for channel_id in shallow_channels:
    passband_low[channel_id] = [1 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [1 * units.MHz, thresholds['fhigh']]
    passband_high[channel_id] = [thresholds['flow'], 800 * units.GHz]
    filter_type[channel_id] = 'butter'
    order_low[channel_id] = 10
    order_high[channel_id] = 5
"""


def task(q, iSim, nu_energy, nu_energy_max, detectordescription, config, output_filename,
         flavor, interaction_type, **kwargs):

    def get_max_radius(E):
        if(E <= 10 ** 16.3 * units.eV):
            return 1.6 * units.km
        elif(E <= 10 ** 16.6 * units.eV):
            return 2.3 * units.km
        elif(E <= 10 ** 17.1 * units.eV):
            return 3.5 * units.km
        elif(E <= 10 ** 17.6 * units.eV):
            return 4.5 * units.km
        elif(E <= 10 ** 18.1 * units.eV):
            return 5.5 * units.km
        elif(E <= 10 ** 18.6 * units.eV):
            return 6.5 * units.km
        elif(E <= 10 ** 19.1 * units.eV):
            return 7 * units.km
        elif(E <= 10 ** 19.6 * units.eV):
            return 8.0 * units.km
        elif(E <= 10 ** 20.1 * units.eV):
            return 8.0 * units.km
        else:
            return 8.0 * units.km

    def get_max_depth(E):
        if(E <= 10 ** 16.3 * units.eV):
            return -0.75 * units.km
        elif(E <= 10 ** 16.6 * units.eV):
            return -1.0 * units.km
        elif(E <= 10 ** 17.1 * units.eV):
            return -1.5 * units.km
        elif(E <= 10 ** 17.6 * units.eV):
            return -2.0 * units.km
        elif(E <= 10 ** 18.1 * units.eV):
            return -2.3 * units.km
        elif(E <= 10 ** 18.6 * units.eV):
            return -2.5 * units.km
        elif(E <= 10 ** 19.1 * units.eV):
            return -2.7 * units.km
        elif(E <= 10 ** 19.6 * units.eV):
            return -2.7 * units.km
        elif(E <= 10 ** 20.1 * units.eV):
            return -2.7 * units.km
        else:
            return -2.7 * units.km

    class mySimulation(simulation.simulation):

        def _detector_simulation_filter_amp(self, evt, station, det):
            # here we apply a larger bandpass filter up to 1GHz
            channelBandPassFilter.run(evt, station, det,
                                      passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
            channelBandPassFilter.run(evt, station, det,
                                      passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
    
        def _detector_simulation_trigger(self, evt, station, det):
            # the trigger is calculated on bandpass limited signals, proceduce
            # 1) creat a copy of the station object
            # 2) apply additional lowpass filter
            # 3) calculate new noise RMS for filtered signals
            # 4) calculate trigger
            # 5) set trigger attributes of original station
    
            # 1) creat a copy of the station object
            station_copy = copy.deepcopy(station)
    
            # 2) apply additional lowpass filter
            channelBandPassFilter.run(evt, station_copy, det,
                                    passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)
    
            # 3) calculate new noise RMS for filtered signals
            Vrms_per_channel_copy = copy.deepcopy(self._Vrms_per_channel)
    
            ff = np.linspace(0, 1 * units.GHz, 10000)
            
            for channel_id in range(station_copy.get_number_of_channels()):
                filt = channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                        passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)
                filt *= channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                        passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
                filt *= channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                        passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
                bandwidth = np.trapz(np.abs(filt) ** 2, ff)
                # the Vrms scales with the squareroot of the bandwidth
                Vrms_per_channel_copy[station_copy.get_id()][channel_id] *= (bandwidth / self._bandwidth_per_channel[station_copy.get_id()][channel_id]) ** 0.5
                if 0:
                    print(f"channel {channel_id}: bandwidth = {bandwidth/units.MHz:.1f}MHz, new Vrms = {Vrms_per_channel_copy[station.get_id()][channel_id]/units.micro/units.V:.4g}muV")
                    tvrms = np.std(station_copy.get_channel(channel_id).get_trace())
                    print(f"\trealized Vrms = {tvrms/units.micro/units.V:.4g}muV")
    
            
            
            # 4) calculate trigger
            # start with the SHALLOW TRIGGER which is always there
            # run a high/low trigger on the 4 downward pointing LPDAs
            """
            threshold_high = {}
            threshold_low = {}
            for channel_id in det.get_channel_ids(station_copy.get_id()):
                threshold_high[channel_id] = thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
                threshold_low[channel_id] = -thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
            highLowThreshold.run(evt, station_copy, det,
                                        threshold_high=threshold_high,
                                        threshold_low=threshold_low,
                                        coinc_window=40 * units.ns,
                                        triggered_channels=shallow_channels,  # select the LPDA channels
                                        number_concidences=2,  # 2/4 majority logic
                                        trigger_name='LPDA_2of4_100Hz')
    
            threshold_high = {}
            threshold_low = {}
            for channel_id in det.get_channel_ids(station_copy.get_id()):
                threshold_high[channel_id] = thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
                threshold_low[channel_id] = -thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
            highLowThreshold.run(evt, station_copy, det,
                                        threshold_high=threshold_high,
                                        threshold_low=threshold_low,
                                        coinc_window=40 * units.ns,
                                        triggered_channels=shallow_channels,  # select the LPDA channels
                                        number_concidences=2,  # 2/4 majority logic
                                        trigger_name='LPDA_2of4_10mHz',
                                        set_not_triggered=not station_copy.has_triggered(trigger_name='LPDA_2of4_100Hz')) # only calculate the trigger at a lower threshold if the previous one triggered
            """


            # DEEP TRIGGER
            # check if the station is a hybrid station
            if(station_copy.get_number_of_channels() > 5):
                # get the Vrms of the phased array channels
                Vrms_PA = Vrms_per_channel_copy[station_copy.get_id()][PA_4ch_channels[0]]
                det_channel = det.get_channel(station_copy.get_id(), PA_4ch_channels[0])
                sampling_rate_phased_array = det_channel["trigger_adc_sampling_frequency"]  # the phased array is digitized with a smaller sampling rate
                # run the 8 phased trigger
                # x4 for upsampling
                window_8ant = int(16 * units.ns * sampling_rate_phased_array * 4.0)
                step_8ant = int(8 * units.ns * sampling_rate_phased_array * 4.0)
    
                # run the 4 phased trigger
                # x2 for upsampling
                window_4ant = int(16 * units.ns * sampling_rate_phased_array * 2.0)
                step_4ant = int(8 * units.ns * sampling_rate_phased_array * 2.0)
                
                """
                phasedArrayTrigger.run(evt, station_copy, det,
                                       Vrms=Vrms_PA,
                                       threshold=thresholds_pa['8ch']['100Hz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                       triggered_channels=PA_8ch_channels,
                                       phasing_angles=phasing_angles_8ant,
                                       ref_index=1.75,
                                       trigger_name=f'PA_8channel_100Hz',  # the name of the trigger
                                       trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                       adc_output=f'voltage',  # output in volts
                                       trigger_filter=None,
                                       upsampling_factor=4,
                                       window=window_8ant,
                                       step=step_8ant)
                """

                phasedArrayTrigger.run(evt, station_copy, det,
                                       Vrms=Vrms_PA,
                                       threshold=thresholds_pa['4ch']['100Hz'] * np.power(Vrms_PA, 2.0),
                                       triggered_channels=PA_4ch_channels,
                                       phasing_angles=phasing_angles_4ant,
                                       ref_index=1.75,
                                       trigger_name=f'PA_4channel_100Hz',  # the name of the trigger
                                       trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                       adc_output=f'voltage',  # output in volts
                                       trigger_filter=None,
                                       upsampling_factor=2,
                                       window=window_4ant,
                                       step=step_4ant)
                
                """
                phasedArrayTrigger.run(evt, station_copy, det,
                                       Vrms=Vrms_PA,
                                       threshold=thresholds_pa['8ch']['1mHz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                       triggered_channels=PA_8ch_channels,
                                       phasing_angles=phasing_angles_8ant,
                                       ref_index=1.75,
                                       trigger_name=f'PA_8channel_1mHz',  # the name of the trigger
                                       trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                       adc_output=f'voltage',  # output in volts
                                       trigger_filter=None,
                                       upsampling_factor=4,
                                       window=window_8ant,
                                       step=step_8ant)
                """
    
                phasedArrayTrigger.run(evt, station_copy, det,
                                       Vrms=Vrms_PA,
                                       threshold=thresholds_pa['4ch']['1mHz'] * np.power(Vrms_PA, 2.0),
                                       triggered_channels=PA_4ch_channels,
                                       phasing_angles=phasing_angles_4ant,
                                       ref_index=1.75,
                                       trigger_name=f'PA_4channel_1mHz',  # the name of the trigger
                                       trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                       adc_output=f'voltage',  # output in volts
                                       trigger_filter=None,
                                       upsampling_factor=2,
                                       window=window_4ant,
                                       step=step_4ant)
    
            # 5) set trigger attributes of original station
            for trigger in station_copy.get_triggers().values():
                station.set_trigger(trigger)
                
            # this module cuts the trace to the record length of the detector
            triggerTimeAdjuster.run(evt, station, det)
            # if(station_copy.has_triggered()):
            #     print("TRIGGERED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
    
            
    flavor_ids = {'e': [12, -12],
                 'mu': [14, -14],
                'tau': [16, -16]}

    r_max = get_max_radius(nu_energy)
    z_min = get_max_depth(nu_energy)
    volume = {'fiducial_rmax': r_max,
            'fiducial_rmin': 0 * units.km,
            'fiducial_zmin': z_min,
            'fiducial_zmax': 0
            }

    #n_events = 1e4
    n_events = 1e4

    input_data = generator.generate_eventlist_cylinder("on-the-fly", n_events, nu_energy, nu_energy_max,
                                                    volume,
                                                    thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                                    phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                                    start_event_id=1,
                                                    flavor=flavor_ids[flavor],
                                                    n_events_per_file=None,
                                                    spectrum='log_uniform',
                                                    #deposited=False,
                                                    deposited=True,
                                                    proposal=False,
                                                    proposal_config='SouthPole',
                                                    start_file_id=0,
                                                    log_level=None,
                                                    proposal_kwargs={},
                                                    max_n_events_batch=n_events,
                                                    write_events=False,
                                                    seed=root_seed + iSim,
                                                    interaction_type=interaction_type)

#     with Pool(20) as p:
#         print(p.map(tmp, input_kwargs))

    sim = mySimulation(inputfilename=input_data,
                       outputfilename=output_filename,
                       detectorfile=detectordescription,
                       outputfilenameNuRadioReco=output_filename+".nur",
                       config_file=config,
                       default_detector_station=1001)
    n_trig = sim.run()

    print(f"simulation pass {iSim} with {n_trig} events", flush=True)
    q.put(n_trig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('energy_min', type=float,
                        help='neutrino energy')
    parser.add_argument('energy_max', type=float,
                        help='neutrino energy')
    parser.add_argument('detectordescription', type=str,
                        help='path to file containing the detector description')
    parser.add_argument('config', type=str,
                        help='NuRadioMC yaml config file')
    parser.add_argument('index', type=int,
                        help='a running index to create unitque files')
    parser.add_argument('flavor', type=str,
                        help='the flavor')
    parser.add_argument('interaction_type', type=str,
                        help='interaction type cc, nc or ccnc')

    args = parser.parse_args()

    kwargs = args.__dict__
    kwargs['nu_energy'] = args.energy_min * units.eV
    kwargs['nu_energy_max'] = args.energy_max * units.eV

    filename = os.path.splitext(os.path.basename(__file__))[0]

    output_folder = f"nu_{args.flavor}_{args.interaction_type}"
    output_path = os.path.join(output_folder, args.detectordescription, args.config, filename, f"{np.log10(kwargs['nu_energy']):.2f}eV", f"{args.index:06}")
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)
    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    class myrunner(runner.NuRadioMCRunner):

        # if required override the get_outputfilename function for a custom output file
        def get_outputfilename(self):
            return os.path.join(self.output_path, f"{np.log10(self.kwargs['nu_energy']):.2f}_{self.kwargs['index']:06d}_{self.i_task:06d}.hdf5")

    # start a simulation on two cores with a runtime of 23h
    #r = myrunner(20, task, output_path, max_runtime=3600 * 24 * 4, n_triggers_max=1e5, kwargs=kwargs)
    #r = myrunner(20, task, output_path, max_runtime=3600 * 24 * 4, n_triggers_max=2000, kwargs=kwargs)
    #r = myrunner(20, task, output_path, max_runtime=3600 * 24 * 4, n_triggers_max=10, kwargs=kwargs)
    r = myrunner(20, task, output_path, max_runtime=3600 * 24 * 4, n_triggers_max=1e5, kwargs=kwargs)
    r.run()
