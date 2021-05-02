import essentia.standard as ess
import os
import json
import numpy as np
import soundAnalysis as SA
import soundDownload as SD

os.chdir("/Users/marcoferreira/Now/Programming/Audio DSP/sms-tools-master/workspace/A9")

def extract_features():

    # Instantiate feature extractor
    extractor = ess.LowLevelSpectralExtractor(frameSize=2048, hopSize=256, sampleRate=44100)

    input_dir = "Data"
    descExt = ".json"
    for paths, dnames, fnames in os.walk(input_dir):
        for fname in fnames:       
            if descExt in fname.lower():
                print(f"Computing features for {paths}") 
                sname = os.path.join(paths, fname.split('.')[0] + '.mp3')

                # Loads sound file
                loader = ess.MonoLoader(filename=sname, sampleRate=44100)
                audio = loader()
                audio_filt = audio.copy()

                # Computes sound energy and threshold
                array_energy = np.square(np.abs(audio))
                mean_energy = np.max(array_energy)
                threshold = 0.3 * mean_energy
                hop = 250
                w_size = 1024

                for i, frame in enumerate(ess.FrameGenerator(audio, frameSize = w_size, hopSize = hop, startFromZero=True)):
                    frame_energy = np.square(np.abs(frame))
                    frame_mean_energy = np.mean(frame_energy)
                    if frame_mean_energy < threshold:
                        try:
                            audio_filt[i*hop:i*hop+len(frame)] = np.zeros(shape=(len(frame),))
                        except:
                            remaining_samples = (i*hop+w_size)-len(audio_filt)
                            audio_filt[i*hop:i*hop+remaining_samples] = np.zeros(remaining_samples)

                
                # Extracts features from filtered signal
                hfc_mean = np.mean(extractor(audio_filt)[4])
                mfcc_mean = np.mean(extractor(audio_filt)[5], axis=0)
                zcr_mean = np.mean(extractor(audio_filt)[25])
                inharmonicity_mean = np.mean(extractor(audio_filt)[26])
                low_mean = np.mean(extractor(audio_filt)[16])
                low_midds_mean = np.mean(extractor(audio_filt)[17])
                high_midds_mean = np.mean(extractor(audio_filt)[18])
                high_mean = np.mean(extractor(audio_filt)[19])
                pitch_salience_mean = np.mean(extractor(audio_filt)[8])
                
                # Extract features that are not present in the LowLevelSpectralExtractor
                attack = ess.LogAttackTime()
                attack_time, a_start, a_end = attack(audio_filt)
                lpc = ess.LPC()
                lpc_coeffs, refl = lpc(audio_filt)
                spec_centroid = ess.SpectralCentroidTime()
                centroid = spec_centroid(audio_filt)

                features = {"lowlevel.hfc.mean":[hfc_mean.tolist()], 
                            "lowlevel.mfcc.mean":[mfcc_mean.tolist()],
                            "lowlevel.zerocrossingrate.mean":[zcr_mean.tolist()],
                            "lowlevel.spectral_energyband_low.mean":[low_mean.tolist()],
                            "lowlevel.spectral_energyband_middle_low.mean":[low_midds_mean.tolist()],
                            "lowlevel.spectral_energyband_middle_high.mean":[high_midds_mean.tolist()],
                            "lowlevel.spectral_energyband_high.mean":[high_mean.tolist()],
                            "lowlevel.pitch_salience_mean":[pitch_salience_mean.tolist()],
                            "lowlevel.inharmonicity.mean":[inharmonicity_mean.tolist()],
                            "sfx.logattacktime.mean":[attack_time],
                            "lowlevel.lpc":[lpc_coeffs.tolist()],
                            "lowlevel.spectralcentroid":[centroid]
                            }

                

                # Dump features into JSON file 
                json.dump(features, open(os.path.join(paths, fname), "w"), indent=4)

def cluster_sounds():
    num_clusters = 10
    accuracy = np.array([])
    desc_list = [i for i in range(10)]
    for i in range(num_clusters):  
        accuracy = np.append(accuracy, SA.clusterSounds("Data", nCluster = num_clusters, descInput=[0,1,2,3,4,5,6,7,8,9]))
    acc_mean = np.mean(accuracy)

    print("\n")
    print("\n")
    print(f"Your mean accuracy was {acc_mean}")


if __name__ == '__main__':

    # extract_features()
    cluster_sounds() 