//simple connection test to the mea device using McsUsbNet.dll

using namespace Mcs.Usb; 

// Create an instance of the CMcsUsbNet class and connect to the device
CMcsUsbNet device = new CMcsUsbNet();
device.Connect();

/*
for multiple devices:
CMcsUsbListNet usblist = new CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB);
var entry = usblist.GetUsbListEntry((uint)0);
CMcsUsbNet device = new CMcsUsbNet();
device.Connect(entry);

*/
// Check if the connection was successful
if (device.IsConnected())
{
    Console.WriteLine("Connection to MEA device successful!");
}
else
{
    Console.WriteLine("Failed to connect to MEA device.");
}

//close the connection when done
device.Disconnect();
