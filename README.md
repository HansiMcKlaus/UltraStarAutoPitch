# UltraStarAutoPitch

Automatically pitches timed lyrics for [UltraStar Deluxe](https://usdx.eu/) using [SPICE](https://tfhub.dev/google/spice/2).



## Dependencies

Install python dependencies: `pip install -r requirements.txt`

This script requires [ffmpeg](https://ffmpeg.org/download.html).

  - Linux:
    - Debian/Ubuntu: `sudo apt-get install ffmpeg`
    - Arch: `sudo pacman -S ffmpeg`
  - macOS: `homebrew install ffmpeg`
  - Windows:
    1. [Download ffmpeg](https://ffmpeg.org/download.html)
    2. Extract it into a folder, for example `C:\FFmpeg`
    3. Add the ffmpeg bin folder to your PATH Environment Variable.
    
    [Here](https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10) is a guide that explains the process in detail.



## Usage

Open a command line and change into the directory where the program is located. It is easiest to simply copy the karaoke and audio file into the same directory, however not at all necessary.

To run the program: `python main.py <Path to karaoke file> <Path to audio file>`

Example: `python main.py '.\Konomi Suzuki - Bursty Greedy Spider.txt' '.\Konomi Suzuki - Bursty Greedy Spider.mp3'`

This creates a copy of `Konomi Suzuki - Bursty Greedy Spider.txt` called `Konomi Suzuki - Bursty Greedy Spider_pitched.txt`.

It is recommended to isolate the vocals as much as possible for better results. Either isolate the vocals using programs like Audacity [via an instrumental version](https://manual.audacityteam.org/man/tutorial_vocal_removal_and_isolation.html#isolate) or use an online service like [vocalremover.org](https://vocalremover.org/) to only use the vocal track of a song.

If SPICE couldn't confidently detect a pitch for a given syllable, it defaults back to 0 (C4). Please go through the file manually to correct errors and missing pitches.



## Flags

`-c, --confidence` How confident the model has to be. Default: 0.85

`-gpu, --gpu` Use GPU instead of CPU. Default: False
