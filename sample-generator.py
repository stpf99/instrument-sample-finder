import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
import os

class SampleGenerator:
    def __init__(self, min_duration_ms=50, max_duration_ms=3000, target_duration_ms=2500):
        self.min_duration = min_duration_ms / 1000
        self.max_duration = max_duration_ms / 1000
        self.target_duration = target_duration_ms / 1000
        
    def load_audio(self, file_path):
        self.audio, self.sr = librosa.load(file_path, sr=None)
        
    def detect_onsets(self):
        # Używamy multiple onset detection functions
        onset_env = librosa.onset.onset_strength(
            y=self.audio, 
            sr=self.sr,
            aggregate=np.median,  # Użycie mediany zamiast średniej dla lepszej odporności
            feature=librosa.feature.melspectrogram,
            n_mels=128
        )
        
        # Adaptacyjne progowanie dla lepszego wykrywania początków
        self.onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            wait=int(self.sr * self.min_duration),
            pre_max=int(self.sr * 0.03),
            post_max=int(self.sr * 0.03),
            pre_avg=int(self.sr * 0.05),
            post_avg=int(self.sr * 0.05),
            delta=0.5,  # Adaptacyjny próg
            backtrack=True  # Dokładniejsze określenie początku
        )

    def analyze_spectral_features(self, audio_segment):
        """Analiza cech spektralnych."""
        if len(audio_segment) < 512:
            return None
            
        # Obliczanie spektrogramu
        D = librosa.stft(audio_segment)
        mag_db = librosa.amplitude_to_db(np.abs(D))
        
        # Cechy spektralne
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_segment))
        
        # Analiza stabilności częstotliwościowej
        freq_stability = np.std(np.mean(mag_db, axis=1))
        
        return {
            'centroid': spectral_centroid,
            'rolloff': spectral_rolloff,
            'flatness': spectral_flatness,
            'stability': freq_stability
        }

    def analyze_temporal_features(self, audio_segment):
        """Analiza cech czasowych."""
        if len(audio_segment) < 512:
            return None
            
        # RMS energy
        rms = librosa.feature.rms(y=audio_segment)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
        
        # Envelope analysis
        envelope = np.abs(hilbert(audio_segment))
        env_stats = {
            'kurtosis': kurtosis(envelope),
            'skewness': skew(envelope),
            'crest_factor': np.max(np.abs(audio_segment)) / np.sqrt(np.mean(np.square(audio_segment)))
        }
        
        return {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'env_stats': env_stats
        }

    def analyze_harmonics(self, audio_segment):
        """Rozszerzona analiza harmoniczna."""
        if len(audio_segment) < 512:
            return 0
        
        # Separacja harmoniczna/perkusyjna
        harmonic, percussive = librosa.effects.hpss(audio_segment)
        
        # Analiza harmoniczności
        harmonic_ratio = np.mean(np.abs(harmonic)) / np.mean(np.abs(audio_segment))
        
        # Pitch salience
        pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=self.sr)
        pitch_salience = np.mean(np.max(magnitudes, axis=0))
        
        # F0 stability
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_segment, 
                                                    fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'))
        f0_stability = 1.0 - np.std(f0[~np.isnan(f0)]) / np.mean(f0[~np.isnan(f0)]) if len(f0[~np.isnan(f0)]) > 0 else 0
        
        return {
            'harmonic_ratio': harmonic_ratio,
            'pitch_salience': pitch_salience,
            'f0_stability': f0_stability,
            'voiced_percentage': np.mean(voiced_flag)
        }

    def is_musical_instrument(self, audio_segment):
        """Rozszerzona weryfikacja czy segment nadaje się na sampel instrumentu."""
        if len(audio_segment) < 512:
            return False
            
        # Analiza cech
        spectral = self.analyze_spectral_features(audio_segment)
        temporal = self.analyze_temporal_features(audio_segment)
        harmonic = self.analyze_harmonics(audio_segment)
        
        if not all([spectral, temporal, harmonic]):
            return False
            
        # Kryteria oceny
        criteria = {
            # Stabilność częstotliwościowa
            'freq_stable': spectral['stability'] < 15,
            
            # Harmoniczność
            'harmonic': harmonic['harmonic_ratio'] > 0.4,
            
            # Stabilność wysokości dźwięku
            'pitch_stable': harmonic['f0_stability'] > 0.7,
            
            # Odpowiednia charakterystyka obwiedni
            'env_good': (temporal['env_stats']['kurtosis'] > 1.5 and 
                        abs(temporal['env_stats']['skewness']) < 2),
            
            # Odpowiedni współczynnik szczytu
            'crest_ok': temporal['env_stats']['crest_factor'] < 10,
            
            # Spójność spektralna
            'spectral_coherent': (spectral['flatness'] < 0.3 and 
                                spectral['centroid'] > 500 and 
                                spectral['centroid'] < 8000)
        }
        
        # Sample musi spełniać większość kryteriów
        return sum(criteria.values()) >= 4

    def normalize_duration(self, audio_segment):
        """
        Normalizacja długości sampla do zadanej długości.
        Metody:
        1. Padding - dodawanie ciszy
        2. Powtarzanie fragmentu
        3. Interpolacja
        """
        current_duration = len(audio_segment) / self.sr
        
        if current_duration >= self.target_duration:
            return audio_segment[:int(self.target_duration * self.sr)]
        
        # Metoda padding
        padding_length = int((self.target_duration - current_duration) * self.sr)
        padded_segment = np.pad(
            audio_segment, 
            (0, padding_length), 
            mode='constant', 
            constant_values=0
        )
        
        return padded_segment
    
    def normalize_amplitude(self, audio_segment, target_db=-12):
        """
        Normalizacja głośności do stałego poziomu decybelowego.
        
        Args:
            audio_segment: Segment audio
            target_db: Docelowy poziom głośności w decybelach
        
        Returns:
            Znormalizowany segment audio
        """
        # Obliczanie bieżącego poziomu RMS
        rms = np.sqrt(np.mean(audio_segment**2))
        current_db = 20 * np.log10(rms)
        
        # Współczynnik wzmocnienia
        gain_factor = 10**((target_db - current_db) / 20)
        
        # Normalizacja z zachowaniem dynamiki
        normalized_segment = audio_segment * gain_factor
        
        # Clip do zakresu [-1, 1]
        normalized_segment = np.clip(normalized_segment, -1, 1)
        
        return normalized_segment

    def advanced_sample_detection(self, audio):
        """
        Zaawansowane wykrywanie sampli z uwzgędnieniem różnych charakterystyk.
        """
        # Wielowarstwowe wykrywanie początków dźwięku
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Próg dynamiczny
        onset_thresh = np.median(onset_env)
        
        # Wykrywanie początków
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=self.sr, 
            units='samples', 
            hop_length=512
        )
        
        # Filtracja na podstawie energii
        filtered_onsets = []
        for onset in onsets:
            window_start = max(0, onset - int(0.1 * self.sr))
            window_end = min(len(audio), onset + int(0.1 * self.sr))
            window = audio[window_start:window_end]
            
            # Kryteria filtracji
            rms_energy = np.sqrt(np.mean(window**2))
            
            if rms_energy > np.percentile(np.abs(audio), 70):
                filtered_onsets.append(onset)
        
        return sorted(filtered_onsets)
    
    def is_interesting_segment(self, segment):
        """
        Ocenia czy segment jest interesujący jako sample.
        """
        if len(segment) < 512:
            return False
        
        # Analiza widmowa
        spectrum = np.abs(np.fft.rfft(segment))
        spectral_centroid = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum)
        
        # Analiza harmoniczności
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        periodicity = np.max(autocorr[1:]) / autocorr[0]
        
        # Analiza dynamiki
        rms = np.sqrt(np.mean(segment**2))
        peak_ratio = np.max(np.abs(segment)) / rms
        
        # Definicja kryteriów
        criteria = [
            spectral_centroid > 500,  
            spectral_centroid < 8000,
            periodicity > 0.5,  
            peak_ratio < 10,    
            rms > np.percentile(np.abs(self.audio), 25)
        ]
        
        return sum(criteria) >= 3
    
    def extract_samples(self, output_dir):
        """Zaawansowane wyodrębnianie sampli."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Zaawansowane wykrywanie początków
        onsets = self.advanced_sample_detection(self.audio)
        
        samples_found = 0
        for i in range(len(onsets) - 1):
            start_idx = onsets[i]
            end_idx = onsets[i + 1]
            duration = (end_idx - start_idx) / self.sr
            
            if self.min_duration <= duration <= self.max_duration:
                segment = self.audio[start_idx:end_idx]
                
                # Dodatkowa weryfikacja segmentu
                if self.is_interesting_segment(segment):
                    # Normalizacje 
                    segment = librosa.util.normalize(segment)
                    segment = np.pad(
                        segment, 
                        (0, int((self.target_duration - len(segment)/self.sr) * self.sr)), 
                        mode='constant'
                    )
                    
                    # Zapis sampla
                    output_path = os.path.join(output_dir, f'sample_{samples_found:04d}.wav')
                    sf.write(output_path, segment, self.sr)
                    samples_found += 1
        
        return samples_found

def process_audio_file(input_file, output_dir):
    generator = SampleGenerator()
    generator.load_audio(input_file)
    num_samples = generator.extract_samples(output_dir)
    return num_samples

def load_audio(self, file_path):
    """Wczytuje plik audio."""
    self.audio, self.sr = librosa.load(file_path, sr=None)

# Dodanie metody load_audio do klasy
SampleGenerator.load_audio = load_audio

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Użycie: python sample_generator.py <plik_wejściowy> <katalog_wyjściowy>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    num_samples = process_audio_file(input_file, output_dir)
    print(f"Wygenerowano {num_samples} sampli w katalogu {output_dir}")
