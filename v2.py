import clr  # pythonnet
import sys

# Load the MCS USB .NET library
sys.path.append(r"C:\Program Files\Multi Channel Systems\MC_Suite")
clr.AddReference("McsUsbNet")
from McsUsbNet import CMcsUsbListNet, DeviceEnumNet

def connect_to_mea():
    # 1. Search for available devices
    device_list = CMcsUsbListNet()
    device_list.Initialize(DeviceEnumNet.MCS_DEVICE_USB)
    
    if device_list.Count > 0:
        # 2. Get the IFB-C Interface Board
        device_info = device_list.GetDeviceInfoByIndex(0)
        print(f"Connecting to: {device_info.DeviceName}")
        
        # 3. Create a device instance (CInterfaceBoardNet for IFB-C)
        # Note: Depending on your specific headstage, you may use CMeaDeviceNet
        from McsUsbNet import CInterfaceBoardNet
        device = CInterfaceBoardNet()
        device.Connect(device_info)
        return device
    else:
        print("No MEA2100-Mini found.")
        return None