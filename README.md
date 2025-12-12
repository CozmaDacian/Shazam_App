An application that tries
to identify the artist of a song given an audio.

The application uses the library librosa
to load the song and then it uses manually created 
functions to process it.

 Processing the data
 First we use the short time fourier to
 convert from the time domain to the frequency domain.
 As we use a sliding window for each time frame
 we also use a Hann Window to smooth out the edges and 
 to not have frequencies that appear due to
 a sudden interuption of the signal.

We create a databse with every artist and every song
and then we begin the hashing process.

Hashing process:
We try to find the highest peaks ( we can also
convert to the mel scale for this process
as the human ear is modeled more to this scale
than decibels). As we usually only get a fragment of a song
we can t just hash that at time x we had y frequency as
the song could be shifted, so we need to 
to hash the highest peaks from each 
each neighbourhood in a 32 bits integr the
first 12 bits represend the anplitude of the
first peak, the next 12 the amplitude of the second and the last
8 bits represent the difference in time between the 2 peaks
as this eliminates the problem with the shifting.
and then we insert this into the databse. To reduce the number of keys per song 
we should modify the neighborhoud constant from the constant files, because if we have
to many keys the retrieval would be extremely slow.
Matching Process:
After we compute the hashes for a song
we have to do a voting process so we use the same hashing alghorithm
. As many song minght have the same hash, we have
to take the song with the highest number of votes as that one is the most likely to be the correct song


Setup: download the ffa small database from github
and then run the populate database which also populates the
the fingerpint table.

