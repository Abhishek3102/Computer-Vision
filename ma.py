import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Get the default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the current volume range
volRange = volume.GetVolumeRange()  # This returns a tuple like (-65.25, 0.0, 0.03125)
minVol = volRange[0]
maxVol = volRange[1]

print(f"Current Volume Range: {volRange}")

# Set volume to minimum
print("Setting volume to minimum...")
volume.SetMasterVolumeLevel(minVol, None)
time.sleep(2)

# Set volume to maximum
print("Setting volume to maximum...")
volume.SetMasterVolumeLevel(maxVol, None)
time.sleep(2)

# Set volume to 50% of the range
vol50 = (maxVol + minVol) / 2
print("Setting volume to 50%...")
volume.SetMasterVolumeLevel(vol50, None)
