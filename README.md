
Shazam_App
This application attempts to identify the artist of a song from a short audio clip.
It uses librosa to load audio and several manually implemented signal-processing functions to extract fingerprints for matching.

How the Processing Works
1. Short-Time Fourier Transform (STFT)
The audio signal is first converted from the time domain to the frequency domain using the Short-Time Fourier Transform.
This allows analyzing the frequency content in short frames instead of the entire audio at once.

2. Hann Window
Because STFT processes audio in overlapping windows, we apply a Hann window to smooth the edges of each frame.
This prevents artificial frequencies from appearing due to sudden discontinuities.
Reference:
https://en.wikipedia.org/wiki/Hann_function

Database Structure
A local database is created that stores:

Artists

Songs

Fingerprints (hashes)

This database is used for fast lookup during the matching phase.

Hashing Process
The fingerprinting method is based on extracting the strongest spectral peaks from each time–frequency region.

Steps:

For each neighborhood of the spectrogram, find the highest peaks
(optionally convert to the mel scale, which is closer to human hearing).

Since the audio fragment may start at an arbitrary time, we cannot hash
“at time X → frequency Y”.
Instead, we use paired peak hashing.
Each hash is stored in a 32-bit integer:

First 12 bits: amplitude of the first peak

Next 12 bits: amplitude of the second peak

Last 8 bits: time difference between the peaks

This eliminates timing-shift problems and makes fingerprints robust even for small clips.

Hashes are inserted into the database.

Important:
If the neighborhood window is too small, you will generate too many hashes, making lookup slow.
This can be tuned in the constants file.

Matching Process

After generating hashes for a query sample:

The same hashing algorithm is applied again, but with more lenient conditions to produce more candidate hashes.

Many songs may share individual hashes, so the system performs a voting process.

The song with the largest number of matching hashes wins — it is considered the most likely match.

Setup

Download the FFA small database from GitHub.

Run the populate database script.
This fills both the song metadata and the fingerprint table.

Machine Learning Component

The project also contains a machine learning module:

PCA (Principal Component Analysis)

A PCA algorithm, implemented from scratch, is used to project audio features into a new space where classification becomes easier.

Current accuracy: ~60%.

Future Improvement

Training an autoencoder for the STFT data would provide better feature compression and reconstruction, improving ML accuracy compared to raw PCA.

To Add

A more user-friendly menu for loading and recognizing audio files.
