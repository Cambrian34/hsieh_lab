import clr
import os
import ctypes
import numpy as np
from collections import deque

from System import Int32, UInt32, Int32, UInt64, Array

# Import the helper you uploaded to convert .NET arrays to NumPy
from clr_array_to_numpy import asNumpyArray

# Load the Multi Channel Systems DLL
clr.AddReference(os.getcwd() + r'\..\..\McsUsbNet\x64\McsUsbNet.dll')
from Mcs.Usb import CMcsUsbListNet, DeviceEnumNet
from Mcs.Usb import CMeaDeviceNet, McsBusTypeEnumNet, DataModeEnumNet, SampleSizeNet, SampleDstSizeNet
from Mcs.Usb import CStg200xDownloadNet, STG_DestinationEnumNet

class MEA2100_OrganoidBrain:
    def __init__(self, sampling_rate=20000):
        self.sampling_rate = sampling_rate
        self.callbackThreshold = self.sampling_rate // 10
        
        # Thread-safe buffer to hold the latest recordings from the OnChannelData event
        self.recent_neural_data = deque(maxlen=5) 
        
        deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        if deviceList.Count == 0:
            raise RuntimeError("No MCS USB devices found. Is the MEA2100 plugged in and powered on?")
        
        listEntry = deviceList.GetUsbListEntry(0)
        print(f"Connecting to Device: {listEntry.DeviceName} Serial: {listEntry.SerialNumber}")

        # --- 1. Setup Recording ---
        self.mea = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS)
        self.mea.ChannelDataEvent += self.OnChannelData
        self.mea.Connect(listEntry)
        
        self.mea.SetSamplerate(self.sampling_rate, 1, 0)
        self.mea.SetDataMode(DataModeEnumNet.Unsigned_16bit, 0)
        
        block = self.mea.GetChannelsInBlock(0)
        mChannels = self.mea.GetChannelsInBlock(0)
        self.mea.ChannelBlock.SetSelectedData(
            mChannels, 
            self.callbackThreshold * 10, 
            self.callbackThreshold, 
            SampleSizeNet.SampleSize16Unsigned, 
            SampleDstSizeNet.SampleDstSize16, 
            block
        )
        self.mea.StartDacq()

        # --- 2. Setup Stimulation ---
        # (Assuming the STG is accessible via the same listEntry. Adjust index if it's a separate USB device)
        self.stg = CStg200xDownloadNet()
        self.stg.Connect(listEntry)
        self.stg.SetVoltageMode()

    def OnChannelData(self, x, cbHandle, numSamples):
        """Asynchronous callback triggered when the MEA has new data."""
        data, frames_ret = self.mea.ChannelBlock.ReadFramesUI16(0, 0, self.callbackThreshold, Int32(0))
        np_data = asNumpyArray(data, ctypes.c_uint16)
        
        # Store the numpy array so the Pygame loop can read it in predict()
        self.recent_neural_data.append(np_data)

    def predict(self, inputs):
        """
        This replaces the LSTM. It takes the game state, stimulates the organoid, 
        reads the reaction, and outputs a movement command.
        """
        relative_angle = inputs[0] # Ranges from -1.0 (Target is Left) to 1.0 (Target is Right)

        # 
        # STEP A: ENCODING (Translatng game angle to stimulation)
        # If target is left, stimulate electrode group 1. If right, group 2.
        channelmap = Array[UInt32]([1,0,0,0])
        syncoutmap = Array[UInt32]([1,0,0,0])
        repeat = Array[UInt32]([1,0,0,0])
        
        # Modify amplitude based on the severity of the angle
        voltage = int(abs(relative_angle) * 100) 
        amplitude = Array[Int32]([-voltage, voltage])
        duration = Array[UInt64]([10000, 10000])

        self.stg.SetupTrigger(0, channelmap, syncoutmap, repeat)
        self.stg.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        self.stg.SendStart(1)

        # ---------------------------------------------------------
        # STEP B: DECODING (Translate MEA recording to game turn)
        # ---------------------------------------------------------
        turn_request = 0.0 
        
        if len(self.recent_neural_data) > 0:
            latest_recording = self.recent_neural_data[-1]
            
            # TODO: Implement your Spike Sorting / Analysis here.
            # You need to identify which array indices correspond to your "Left Motor" 
            # and "Right Motor" electrodes, and calculate a turn_request based on activity.
            
            # Example Placeholder Logic:
            # left_activity = np.sum(latest_recording[0:10])
            # right_activity = np.sum(latest_recording[10:20])
            # turn_request = (right_activity - left_activity) * scaling_factor
            pass

        return [turn_request]

    def cleanup(self):
        """Gracefully disconnect to prevent locking up the hardware."""
        self.mea.StopDacq()
        self.mea.Disconnect()
        self.stg.Disconnect()