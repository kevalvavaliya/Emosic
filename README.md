## Inspiration💡💡
We always play music according to our mood so we created an application so we can sort the playlist for the moods and the app will play it automatically, We need to express ourselves to the camera and it will pick the music according to our mood.

## What it does
The app takes the playlist and sorts it according to the mood and music we like when we feel happy or sad.
An ML model uses a live camera feed to detect the mood from our facial expressions and will play the song from the playlist accordingly.

The app determines what was the longest period of mood detected in the last 5 seconds and will change the track accordingly.

## How we built it
We have used the TensorFlow FER2013 data set from Kagel to train the model,  we are using flask to give users an interface accessible from everywhere to manage their playlist.
the app takes permission for player control with Oauth 2.0 and changes the track according to the mood detected in the ML model.

## Challenges we ran into
Integrating the model in the backend and using it for face detection and sending the mood at the best time was bit difficult as model detects the mood in every single frame of the video captured and it facel expression changes every micro seconds so it may change the track so frequently that user will be annoyed or spotify API will hit rate limit.

## Accomplishments that we're proud of
We completed the project in a short period of time and We integrated ML with the backend.
It was the first time using Spotify API and we created awsome project with it.
