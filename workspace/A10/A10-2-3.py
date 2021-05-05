import essentia.standard as ess
import os
import json
import numpy as np
import soundAnalysis as SA

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
                mean_energy = np.mean(array_energy)
                threshold = 0.2 * mean_energy
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
                # hfc_mean = np.mean(extractor(audio_filt)[4])
                mfcc_mean = np.mean(extractor(audio_filt)[5], axis=0, dtype=float)
                zcr_mean = np.mean(extractor(audio_filt)[25],dtype=float)
                inharmonicity_mean = np.mean(extractor(audio_filt)[26], dtype=float)
                # low_mean = np.mean(extractor(audio_filt)[16], dtype=float)
                # low_midds_mean = np.mean(extractor(audio_filt)[17], dtype=float)
                # high_midds_mean = np.mean(extractor(audio_filt)[18], dtype=float)
                # high_mean = np.mean(extractor(audio_filt)[19], dtype=float)
                # pitch_salience_mean = np.mean(extractor(audio_filt)[8])
                
                # Extract features that are not present in the LowLevelSpectralExtractor
                attack = ess.LogAttackTime()
                attack_time, a_start, a_end = attack(audio_filt)
                # lpc = ess.LPC()
                # lpc_coeffs, refl = lpc(audio_filt)
                spec_centroid = ess.SpectralCentroidTime()
                centroid = float(spec_centroid(audio_filt))

                # features = {"lowlevel.hfc.mean":[hfc_mean.tolist()], 
                #             "lowlevel.mfcc.mean":[mfcc_mean.tolist()],
                #             "lowlevel.zerocrossingrate.mean":[zcr_mean.tolist()],
                #             "lowlevel.spectral_energyband_low.mean":[low_mean.tolist()],
                #             "lowlevel.spectral_energyband_middle_low.mean":[low_midds_mean.tolist()],
                #             "lowlevel.spectral_energyband_middle_high.mean":[high_midds_mean.tolist()],
                #             "lowlevel.spectral_energyband_high.mean":[high_mean.tolist()],
                #             "lowlevel.pitch_salience_mean":[pitch_salience_mean.tolist()],
                #             "lowlevel.inharmonicity.mean":[inharmonicity_mean.tolist()],
                #             "sfx.logattacktime.mean":[attack_time],
                #             "lowlevel.lpc":[lpc_coeffs.tolist()],
                #             "lowlevel.spectralcentroid":[centroid]
                #             }
                features = {"lowlevel.mfcc.mean.0":[mfcc_mean[0]],
                "lowlevel.mfcc.mean.1":[mfcc_mean[1]],
                "lowlevel.mfcc.mean.2":[mfcc_mean[2]],
                "lowlevel.mfcc.mean.3":[mfcc_mean[3]],
                "lowlevel.mfcc.mean.4":[mfcc_mean[4]],
                "lowlevel.mfcc.mean.5":[mfcc_mean[5]],
                "lowlevel.mfcc.mean.6":[mfcc_mean[6]],
                "lowlevel.mfcc.mean.7":[mfcc_mean[7]],
                "lowlevel.mfcc.mean.8":[mfcc_mean[8]],
                "lowlevel.mfcc.mean.9":[mfcc_mean[9]],
                "lowlevel.mfcc.mean.10":[mfcc_mean[10]],
                "lowlevel.mfcc.mean.11":[mfcc_mean[11]],
                "lowlevel.mfcc.mean.12":[mfcc_mean[12]],
                "sfx.logattacktime.mean":[attack_time],
                "lowlevel.inharmonicity.mean":[inharmonicity_mean],
                "lowlevel.zerocrossingrate.mean":[zcr_mean],
                "lowlevel.spectralcentroid":[centroid]
                }

                # Dump features into JSON file 
                json.dump(features, open(os.path.join(paths, fname), "w"), indent=4)

def cluster_sounds():
    num_clusters = 10
    accuracy = np.array([])
    desc_list = [i for i in range(10)]
    for i in range(num_clusters):  
        accuracy = np.append(accuracy, SA.clusterSounds("Data", nCluster = num_clusters, descInput=np.arange(0,18)))
    acc_mean = np.mean(accuracy)

    print("\n")
    print("\n")
    print(f"Your mean accuracy is {acc_mean}")


if __name__ == '__main__':

    # extract_features()
    cluster_sounds() 