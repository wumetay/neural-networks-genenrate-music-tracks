import os
from pydub import AudioSegment
out_path = "./song/wav/"
def mp3_to_wav(path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    os.chdir(path)
    audio_files = os.listdir(path)

    for file in audio_files:

        name, ext = os.path.splitext(file)
        if ext == ".mp3":
           mp3_sound = AudioSegment.from_mp3(file)
           mp3_sound.export("{0}.wav".format(out_path), format="wav")

def mp32wav(path):
    global out_path
    
    head,file = os.path.split(path)
    name,ext = os.path.splitext(file)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if ext == ".mp3":
        mp3_sound = AudioSegment.from_mp3(path)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        mp3_sound.export( "{0}/wav/{1}.wav".format(head,name), format="wav")

if __name__ == "__main__":
    mp32wav("song/NoNoNo.mp3")