
# 21.09.2026
# Version 1.0.0.0
# Make_Accessible
 

print("START", flush=True)
import json
import time
start = time.time()
import argparse
parser = argparse.ArgumentParser(description="Process a PDF file.")
parser.add_argument('-i', '--input', required=True, help='Path to input PDF file')
parser.add_argument('-o', '--output', required=True, help='Path to output PDF file')
args = parser.parse_args()

inpdf = args.input
outpdf = args.output

print("Loading Pdfix...", flush=True)
from pdfixsdk import *

print("Loading Utils...", flush=True)
import Utils

commandPath = ""  # inputPath + "/make-accessible.json"

print("GetPdfix...", flush=True)
pdfix = GetPdfix()
if pdfix is None:
    raise Exception("Pdfix Initialization fail")

print("OpenDoc...", flush=True)
doc = pdfix.OpenDoc(inpdf, "")
if doc is None:
    raise Exception("Unable to open pdf : " + pdfix.GetError())

print("GetCommand...", flush=True)
command = doc.GetCommand()
if command is None:
    raise Exception(pdfix.GetError())

cmdStm = None


def extract_json_name(json_text):
    if not json_text:
        return None

    try:
        data = json.loads(json_text)
        return data.get("name")
    except Exception:
        return None


try:
    # load the make_accessible command from JSON file
    # or find the embedded custom action named "make_accessible"
    # print("commandPath : ",commandPath)
    if commandPath == "":
        cmd_count = command.GetNumCustomActions()

        for i in range(cmd_count):
            tmpStm = pdfix.CreateMemStream()
            if tmpStm is None:
                raise Exception(pdfix.GetError())

            try:
                if not command.SaveCustomActionToStream(
                    i, tmpStm, kDataFormatJson, kSaveFull
                ):
                    raise Exception(pdfix.GetError())

                json_text = bytearray(Utils.stream_to_data(tmpStm))
                
                text = json_text.decode("utf-8")
                daten = json.loads(text)

                name = extract_json_name(json_text)
                # print("Name JSON : ",name)

                if name == "make_accessible":
                    print("Name JSON make_accessible gefunden : ",name)
                    cmdStm = tmpStm
                    tmpStm = None
                    break
            except Exception as e:
                print("[WARNING]: Loading JSON command failed. [{}] {}".format(i, e))
            finally:
                if tmpStm is not None:
                    tmpStm.Destroy()

        if cmdStm is None:
            raise Exception("Embedded custom action 'make_accessible' was not found.")
    else:
        cmdStm = pdfix.CreateFileStream(commandPath, kPsReadOnly)
        if cmdStm is None:
            raise Exception(pdfix.GetError())

    if not command.LoadParamsFromStream(cmdStm, kDataFormatJson):
        raise Exception(pdfix.GetError())

    cmdStm.Destroy()
    cmdStm = None

    # run the command
    # https://docs.pdfix.net/sdk/api/pdfix/ps-command/
    print("Running command...", flush=True)
    if not command.Run():
        raise Exception(pdfix.GetError())

    print("Save...", flush=True)
    if not doc.Save(outpdf, kSaveFull):
        raise Exception(pdfix.GetError())
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    raise    

finally:
    if cmdStm is not None:
        cmdStm.Destroy()
    doc.Close()
    
end = time.time()
print("Dauer:", round((end - start), 2), "Sekunden")    