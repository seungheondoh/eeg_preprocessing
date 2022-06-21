DATASET="../../.."
INT_RANDOM_SEED = 42
# MUSIC_SAMPLE_RATE = 22050
MUSIC_SAMPLE_RATE = 16000
STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'
DATA_LENGTH = MUSIC_SAMPLE_RATE * 30
INPUT_LENGTH = MUSIC_SAMPLE_RATE * 3
CHUNK_SIZE = 16

DEAP_CHANNEL =["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
DEP_PERIPHERAL =["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"]
LABELS = ['HAHV','HALV','LAHV','LALV']
BAND = {'theta': [4, 8],'alpha': [8, 12],'beta': [12, 30],'gamma': [30, 64]}
DEAP_Start = int(128 * 3)