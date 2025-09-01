import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class VoiceAgeDataset(Dataset):
    """Custom dataset for voice age detection"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class VoiceAgeDetector(nn.Module):
    """Neural network for voice age detection"""
    def __init__(self, input_size, num_classes):
        super(VoiceAgeDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class VoiceAgeDetectorSystem:
    def __init__(self):
        self.sample_rate = 22050
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.age_groups = {
            'child': (0, 12),
            'teen': (13, 19),
            'young_adult': (20, 35),
            'adult': (36, 55),
            'senior': (56, 100)
        }
    
    def extract_features(self, audio_path):
        """Extract audio features for age detection"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            features = []
            
            # 1. Pitch and fundamental frequency features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitches = pitches[pitches > 0]  # Remove zero values
            if len(pitches) > 0:
                features.append(np.mean(pitches))  # Mean pitch
                features.append(np.std(pitches))   # Pitch standard deviation
                features.append(np.median(pitches)) # Median pitch
            else:
                features.extend([0, 0, 0])  # Default values if no pitch detected
            
            # 2. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # 3. MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features.append(np.mean(mfccs[i]))
                features.append(np.std(mfccs[i]))
            
            # 4. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 5. RMS energy
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # 6. Harmonic features
            harmonic = librosa.effects.harmonic(y=audio)
            features.append(np.mean(harmonic))
            features.append(np.std(harmonic))
            
            # 7. Tempo features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def create_synthetic_dataset(self, num_samples=500):
        """Create a synthetic dataset for demonstration"""
        # In a real project, you would use a real dataset like:
        # - Common Voice (Mozilla)
        # - TIMIT
        # - VoxCeleb
        # - Your own collected data
        
        print("Creating synthetic dataset for demonstration...")
        features = []
        labels = []
        
        # Generate synthetic features for different age groups
        for age_group, (min_age, max_age) in self.age_groups.items():
            for _ in range(num_samples // len(self.age_groups)):
                # Base feature values for each age group
                if age_group == 'child':
                    base_features = [250, 50, 240, 1800, 300, 2800, 400, 1200, 200]
                elif age_group == 'teen':
                    base_features = [200, 45, 190, 1600, 250, 2500, 350, 1100, 180]
                elif age_group == 'young_adult':
                    base_features = [180, 40, 170, 1500, 220, 2300, 320, 1000, 160]
                elif age_group == 'adult':
                    base_features = [160, 35, 150, 1400, 200, 2100, 290, 900, 140]
                else:  # senior
                    base_features = [140, 30, 130, 1300, 180, 1900, 260, 800, 120]
                
                # Add some randomness
                synthetic_features = [f + np.random.normal(0, f*0.1) for f in base_features]
                
                # Add MFCC-like features
                for i in range(26):  # 13 MFCCs * 2 (mean and std)
                    synthetic_features.append(np.random.normal(0, 10))
                
                # Add remaining features
                synthetic_features.extend([
                    np.random.normal(0.1, 0.02),  # ZCR mean
                    np.random.normal(0.05, 0.01), # ZCR std
                    np.random.normal(0.1, 0.02),  # RMS mean
                    np.random.normal(0.03, 0.01), # RMS std
                    np.random.normal(0.8, 0.1),   # Harmonic mean
                    np.random.normal(0.1, 0.02),  # Harmonic std
                    np.random.normal(120, 20)     # Tempo
                ])
                
                features.append(synthetic_features)
                labels.append(age_group)
        
        return np.array(features), np.array(labels)
    
    def train_models(self, features, labels):
        """Train both traditional ML and neural network models"""
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors for neural network
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 1. Train Random Forest
        print("Training Random Forest classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # 2. Train SVM
        print("Training SVM classifier...")
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        svm_pred = svm_model.predict(X_test_scaled)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        
        # 3. Train Neural Network
        print("Training Neural Network...")
        train_dataset = VoiceAgeDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        self.model = VoiceAgeDetector(X_train_scaled.shape[1], len(self.age_groups)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/100], Loss: {total_loss/len(train_loader):.4f}')
        
        # Evaluate neural network
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(self.device)
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            nn_accuracy = accuracy_score(y_test, predicted.cpu().numpy())
            print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
        
        # Print classification report for the best model
        best_accuracy = max(rf_accuracy, svm_accuracy, nn_accuracy)
        if best_accuracy == rf_accuracy:
            print("\nBest model: Random Forest")
            print(classification_report(y_test, rf_pred, target_names=self.label_encoder.classes_))
            self.model = rf_model
            self.model_type = "random_forest"
        elif best_accuracy == svm_accuracy:
            print("\nBest model: SVM")
            print(classification_report(y_test, svm_pred, target_names=self.label_encoder.classes_))
            self.model = svm_model
            self.model_type = "svm"
        else:
            print("\nBest model: Neural Network")
            print(classification_report(y_test, predicted.cpu().numpy(), target_names=self.label_encoder.classes_))
            self.model_type = "neural_network"
        
        return best_accuracy
    
    def predict_age(self, audio_path):
        """Predict the age group of a voice"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # Extract features from audio
        features = self.extract_features(audio_path)
        if features is None:
            print("Failed to extract features from audio.")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        if self.model_type in ["random_forest", "svm"]:
            prediction = self.model.predict(features_scaled)
            age_group = self.label_encoder.inverse_transform(prediction)[0]
        else:  # neural network
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs.data, 1)
                age_group = self.label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        
        return age_group
    
    def analyze_audio(self, audio_path):
        """Comprehensive audio analysis for age detection"""
        print(f"Analyzing audio: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Waveform
        plt.subplot(3, 2, 1)
        plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 2. Spectrogram
        plt.subplot(3, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # 3. Spectral centroid
        plt.subplot(3, 2, 3)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        times = librosa.times_like(spectral_centroid)
        plt.plot(times, spectral_centroid.T)
        plt.title('Spectral Centroid')
        plt.xlabel('Time')
        plt.ylabel('Hz')
        
        # 4. MFCCs
        plt.subplot(3, 2, 4)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCCs')
        
        # 5. Zero crossing rate
        plt.subplot(3, 2, 5)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        plt.plot(times, zcr.T)
        plt.title('Zero Crossing Rate')
        plt.xlabel('Time')
        
        # 6. RMS energy
        plt.subplot(3, 2, 6)
        rms = librosa.feature.rms(y=audio)
        plt.plot(times, rms.T)
        plt.title('RMS Energy')
        plt.xlabel('Time')
        
        plt.tight_layout()
        plt.savefig('audio_analysis.png')
        plt.close()
        
        print("Audio analysis complete. Results saved to 'audio_analysis.png'")
        
        # Extract and display key features
        features = self.extract_features(audio_path)
        if features is not None:
            print("\nKey Audio Features:")
            feature_names = [
                'Mean Pitch', 'Pitch Std', 'Median Pitch',
                'Spectral Centroid Mean', 'Spectral Centroid Std',
                'Spectral Rolloff Mean', 'Spectral Rolloff Std',
                'Spectral Bandwidth Mean', 'Spectral Bandwidth Std'
            ]
            
            for i, (name, value) in enumerate(zip(feature_names, features[:9])):
                print(f"{name}: {value:.2f}")
        
        return features

# Main execution
if __name__ == "__main__":
    # Initialize the system
    detector = VoiceAgeDetectorSystem()
    
    # Create and train on synthetic data
    print("=== Voice Age Detection System ===")
    features, labels = detector.create_synthetic_dataset(num_samples=500)
    accuracy = detector.train_models(features, labels)
    print(f"Overall best accuracy: {accuracy:.4f}")
    
    # Test with a real audio file (replace with your file path)
    test_audio = "your_voice_sample.wav"  # Change this to your audio file
    
    if os.path.exists(test_audio):
        # Analyze the audio
        features = detector.analyze_audio(test_audio)
        
        # Predict age group
        age_group = detector.predict_age(test_audio)
        if age_group:
            min_age, max_age = detector.age_groups[age_group]
            print(f"\nPredicted Age Group: {age_group} ({min_age}-{max_age} years)")
    else:
        print(f"\nTest file {test_audio} not found.")
        print("Please place your audio file in the same directory and update the 'test_audio' variable.")