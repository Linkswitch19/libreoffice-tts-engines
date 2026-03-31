import argparse
import soundfile as sf
#without PyTorch
from kokoro_onnx import Kokoro
import sys
import os






def main():
    args_tts =parser_TTSKokoro()
    text_to_read = args_tts.text
    if not text_to_read:
        #Controlla se si sta usando un terminale o se c'e sul stdin
        if not sys.stdin.isatty():
            text_to_read = sys.stdin.read().strip()
        else:
            print("Errore: No Text found on stdin.")
            sys.exit(1)
    
    model_path,voices_path=extration_path()
    samples, saple_rate=kokoro_engine_generation(text_to_read,args_tts,model_path,voices_path)
    #save file
    sf.write(args_tts.output,samples,saple_rate)
    print(f"Saved in {args_tts.output}")



def kokoro_engine_generation(text,args_tts,model_path,voices_path):
     #start Kokoro
    kokoro_tts= Kokoro(model_path,voices_path)
    
    #generate audio
    print(f"Generate audio for: {text[:30]}...")
    samples, saple_rate = kokoro_tts.create(
        text,
        voice=args_tts.voice,
        speed=1.0,
        lang=args_tts.lang
    )
    return samples, saple_rate
    

def extration_path():
    #check if im using the exe and choose the path 
    if getattr(sys, 'frozen',False):
        #extract the file here
        base_path= sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
    model_path = os.path.join(base_path, "kokoro-v1.0.int8.onnx")
    voices_path = os.path.join(base_path, "voices-v1.0.bin")
    if not os.path.exists(model_path) or not os.path.exists(voices_path):
        print(f"Error: file of model or voice not found in {base_path}")
        sys.exit(1)
    return model_path,voices_path
    
         
def parser_TTSKokoro():
    """
    Parser for to read args of the voice 
    """
    parser = argparse.ArgumentParser(description='Generatore TTS Kokoro ONNX')
    parser.add_argument('--text',type=str,required=False, help="Text to read")
    parser.add_argument('--output',type=str, required=True, help='Path to save .wav')
    parser.add_argument('--lang', type=str, default= 'en-us', help='Language (ex. en-us,it)')
    parser.add_argument('--voice', type=str ,default='af_heart', help = 'Select voice')
    return parser.parse_args()
    
if __name__ == "__main__":
    main()