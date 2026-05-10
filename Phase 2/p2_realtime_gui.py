"""
Phase 2 - Version 2: Real-Time Spike Viewer + Stimulation GUI
Displays real-time neural activity from MEA device with stimulation controls.
Features: multi-channel spike visualization, threshold adjustment, and electrical stimulation buttons.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os
from queue import Queue
from collections import deque
import time

# MEA Device references
try:
    import clr
    clr.AddReference(os.path.dirname(os.path.abspath(__file__)) + r'\..\..\McsUsbNet\x64\McsUsbNet.dll')
    from Mcs.Usb import CMcsUsbListNet, DeviceEnumNet, CMeaDeviceNet, CStg200xDownloadNet
    from Mcs.Usb import McsBusTypeEnumNet, DataModeEnumNet, SampleSizeNet, SampleDstSizeNet
    from Mcs.Usb import STG_DestinationEnumNet
    from System import *
    MEA_AVAILABLE = True
    print("MEA device connected")

except (ImportError, ModuleNotFoundError, Exception):
    MEA_AVAILABLE = False
    print("Warning: MEA device not available")
    print("Warning: MEA device libraries not found. Running in simulation mode.")

class MEADataAcquisition:
    """Handles MEA recording"""
    def __init__(self, data_queue, num_channels=60):
        self.data_queue = data_queue
        self.num_channels = num_channels
        self.device = None
        self.is_recording = False
        self.sampling_rate = 25000
        self.callback_threshold = self.sampling_rate // 20  # 50ms windows
        self.current_data = deque(maxlen=25000)  # 1 second of data
        
    def on_channel_data(self, x, cbHandle, numSamples):
        """Callback for MEA data"""
        try:
            dataArrays, _ = self.device.ChannelBlock.ReadAsFrameArrayI32(
                0, 0, self.callback_threshold, Int32(0)
            )
            data_array = np.asarray(dataArrays)
            # Store and queue data
            self.data_queue.put(data_array)
        except Exception as e:
            print(f"Error in callback: {e}")

    def connect(self):
        """Connect to MEA device"""
        if not MEA_AVAILABLE:
            return False
        try:
            deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
            if deviceList.Count == 0:
                return False
            
            self.device = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS)
            self.device.ChannelDataEvent += self.on_channel_data
            self.device.Connect(deviceList.GetUsbListEntry(0))
            
            self.device.SetSamplerate(self.sampling_rate, 1, 0)
            self.device.SetVoltageRangeByIndex(0, 0)
            self.device.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
            self.device.SetNumberOfChannels(0)
            self.device.SetNumberOfAnalogChannels(self.num_channels, 0, 0, 0, 0)
            self.device.EnableDigitalIn(Boolean(False), UInt32(0))
            self.device.EnableChecksum(False, 0)
            
            block = self.device.GetChannelsInBlock(0)
            mChannels = block // 2
            
            self.device.ChannelBlock.SetSelectedData(
                mChannels, self.sampling_rate * 1, self.callback_threshold,
                SampleSizeNet.SampleSize32Signed, SampleDstSizeNet.SampleDstSize32, block
            )
            return True
        except Exception as e:
            print(f"MEA connection error: {e}")
            return False

    def start(self):
        if self.device:
            self.device.StartDacq()
            self.is_recording = True

    def stop(self):
        if self.device:
            self.device.StopDacq()
            self.is_recording = False

    def disconnect(self):
        if self.device:
            self.stop()
            self.device.Disconnect()

class StimulationController:
    """Handles electrical stimulation via STG200x"""
    def __init__(self):
        self.device = None
        self.is_connected = False
        
    def connect(self):
        """Connect to stimulation device"""
        if not MEA_AVAILABLE:
            return False
        try:
            deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
            if deviceList.Count == 0:
                return False
            
            self.device = CStg200xDownloadNet()
            self.device.Connect(deviceList.GetUsbListEntry(0))
            self.is_connected = True
            
            voltageRange = self.device.GetVoltageRangeInMicroVolt(0)
            currentRange = self.device.GetCurrentRangeInNanoAmp(0)
            print(f"Stimulator connected - Voltage Range: {voltageRange/1000} mV, Current Range: {currentRange/1000} uA")
            return True
        except Exception as e:
            print(f"Stimulation connection error: {e}")
            return False

    def send_pulse(self, channels=[0], amplitude_nA=100000, duration_us=500000, frequency_hz=120):
        """Send a stimulation pulse"""
        if not self.is_connected or not self.device:
            return False
        try:
            # Simple biphasic pulse pattern
            num_pulses = int(frequency_hz * (duration_us / 1000000))
            if num_pulses == 0:
                num_pulses = 1
            pulse_duration = int(duration_us / (num_pulses * 2))
            
            amplitude_list = []
            duration_list = []
            
            for _ in range(num_pulses):
                amplitude_list.append(int(amplitude_nA))
                duration_list.append(pulse_duration)
                amplitude_list.append(-int(amplitude_nA))
                duration_list.append(pulse_duration)
            
            channelmap_list = [0, 0, 0, 0]
            for ch in channels:
                idx = ch // 32
                if idx < 4:
                    channelmap_list[idx] |= (1 << (ch % 32))
            
            channelmap = Array[UInt32](channelmap_list)
            syncoutmap = Array[UInt32]([1, 0, 0, 0])
            repeat = Array[UInt32]([1, 0, 0, 0])
            
            amplitude = Array[Int32](amplitude_list)
            duration = Array[UInt64](duration_list)
            
            self.device.SetupTrigger(0, channelmap, syncoutmap, repeat)
            self.device.SetCurrentMode()
            self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_current)
            self.device.SendStart(1)
            return True
        except Exception as e:
            print(f"Error sending pulse: {e}")
            return False

    def disconnect(self):
        if self.device:
            self.device.Disconnect()

class RealtimeGUI:
    """Main GUI for real-time spike visualization and stimulation"""
    def __init__(self, root):
        self.root = root
        self.root.title("Phase 2: Real-Time Spike Viewer + Stimulation")
        self.root.geometry("1400x800")
        
        # Initialize devices
        self.data_queue = Queue()
        self.mea = MEADataAcquisition(self.data_queue)
        self.stimulator = StimulationController()
        
        self.mea_connected = self.mea.connect()
        self.stim_connected = self.stimulator.connect()
        
        if self.mea_connected:
            self.mea.start()
        else:
            # Start a simulation thread if MEA is not connected
            self.sim_thread = threading.Thread(target=self._simulation_thread, daemon=True)
            self.sim_thread.start()
        
        # Create GUI elements
        self.create_widgets()
        
        # Data for visualization
        self.spike_data = deque(maxlen=2500)  # 100ms at 25kHz
        self.channel_data = {i: deque(maxlen=2500) for i in range(60)}
        self.spike_threshold = 3.0
        
        # Update loop
        self.update_plots()

    def _simulation_thread(self):
        """Generates fake MEA data when hardware is not present."""
        print("Starting data simulation thread...")
        num_channels = 60
        # This is the size of a single data chunk from the callback
        chunk_size = self.mea.callback_threshold 
        
        while True:
            # Generate a chunk of fake data (random noise)
            # The real data is int32, so let's simulate that.
            fake_data = np.random.randint(-2000, 2000, size=(chunk_size, num_channels), dtype=np.int32)
            
            # Add some random spikes to a few channels
            for _ in range(np.random.randint(0, 5)): # 0 to 4 spikes per chunk
                spike_channel = np.random.randint(0, num_channels)
                spike_start = np.random.randint(0, chunk_size - 10)
                spike_amp = np.random.randint(8000, 20000) * (1 if np.random.rand() > 0.5 else -1)
                fake_data[spike_start:spike_start+2, spike_channel] = spike_amp
                fake_data[spike_start+2:spike_start+5, spike_channel] = -spike_amp / 2

            self.data_queue.put(fake_data)
            # The callback is triggered every 50ms
            time.sleep(0.05)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Status indicators
        status_frame = ttk.LabelFrame(control_frame, text="Status")
        status_frame.pack(side=tk.LEFT, padx=5)
        
        self.mea_label = ttk.Label(status_frame, text=f"MEA: {'Connected' if self.mea_connected else 'Disconnected'}", 
                                   foreground='green' if self.mea_connected else 'red')
        self.mea_label.pack()
        
        self.stim_label = ttk.Label(status_frame, text=f"Stimulator: {'Connected' if self.stim_connected else 'Disconnected'}", 
                                    foreground='green' if self.stim_connected else 'red')
        self.stim_label.pack()
        
        # Stimulation controls
        stim_frame = ttk.LabelFrame(control_frame, text="Stimulation Controls")
        stim_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Label(stim_frame, text="Electrodes (0-59, e.g., 0, 1-5, all):").pack(side=tk.LEFT, padx=5)
        self.electrodes_var = tk.StringVar(value="0")
        electrodes_entry = ttk.Entry(stim_frame, textvariable=self.electrodes_var, width=15)
        electrodes_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(stim_frame, text="Amp (µA):").pack(side=tk.LEFT, padx=5)
        self.amplitude_var = tk.StringVar(value="100")
        amplitude_spin = ttk.Spinbox(stim_frame, from_=5, to=140, textvariable=self.amplitude_var, width=5)
        amplitude_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stim_frame, text="Dur (ms):").pack(side=tk.LEFT, padx=5)
        self.duration_var = tk.StringVar(value="500")
        duration_spin = ttk.Spinbox(stim_frame, from_=50, to=1000, textvariable=self.duration_var, width=5)
        duration_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stim_frame, text="Freq (Hz):").pack(side=tk.LEFT, padx=5)
        self.freq_var = tk.StringVar(value="120")
        freq_spin = ttk.Spinbox(stim_frame, from_=10, to=500, textvariable=self.freq_var, width=5)
        freq_spin.pack(side=tk.LEFT, padx=5)
        
        self.pulse_button = ttk.Button(stim_frame, text="Send Pulse", command=self.send_pulse)
        self.pulse_button.pack(side=tk.LEFT, padx=10)
        
        # Threshold adjustment
        thresh_frame = ttk.LabelFrame(control_frame, text="Spike Threshold (σ)")
        thresh_frame.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=3.0)
        threshold_scale = ttk.Scale(thresh_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                    variable=self.threshold_var, command=self.update_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = ttk.Label(thresh_frame, text="3.0σ")
        self.threshold_label.pack(side=tk.LEFT)
        
        # Plotting area
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(14, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots for different channels
        self.axes = [self.fig.add_subplot(6, 10, i+1) for i in range(60)]
        
    def update_threshold(self, value):
        """Update spike threshold"""
        self.spike_threshold = float(value)
        self.threshold_label.config(text=f"{self.spike_threshold:.1f}σ")
        
    def send_pulse(self):
        """Send stimulation pulse"""
        if not self.stim_connected:
            messagebox.showwarning("Warning", "Stimulator not connected (running in simulation)")
            # In simulation mode we just show a success message to indicate it worked in logic
            # return

        try:
            amplitude = int(self.amplitude_var.get()) * 1000  # Convert to nanoamps
            duration = int(self.duration_var.get()) * 1000    # Convert to microseconds
            frequency = int(self.freq_var.get())
            
            # Parse electrodes
            electrodes_str = self.electrodes_var.get().strip().lower()
            channels = set()
            if electrodes_str == 'all':
                channels = set(range(60))
            else:
                for part in electrodes_str.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        channels.update(range(start, end + 1))
                    else:
                        channels.add(int(part))
            channels = [ch for ch in channels if 0 <= ch < 60]
            if not channels:
                raise ValueError("No valid channels selected. Please select 0-59.")
            
            if self.stim_connected:
                if self.stimulator.send_pulse(channels, amplitude, duration, frequency):
                    messagebox.showinfo("Success", f"Pulse sent to {len(channels)} channels: {amplitude/1000}µA, {duration/1000}ms, {frequency}Hz")
                else:
                    messagebox.showerror("Error", "Failed to send pulse")
            else:
                # Simulation success
                print(f"[Simulation] Pulse sent to {len(channels)} channels ({channels}): {amplitude/1000}µA, {duration/1000}ms, {frequency}Hz")
                messagebox.showinfo("Success", f"[Sim] Pulse sent to {len(channels)} channels: {amplitude/1000}µA, {duration/1000}ms, {frequency}Hz")

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter values: {e}")
    
    def update_plots(self):
        """Update plots with new data"""
        # Get data from queue
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                # data shape should be (chunk_size, num_channels)
                data_T = data.T
                for i, channel_timeseries in enumerate(data_T):
                    if i < 60:
                        self.channel_data[i].extend(channel_timeseries)
            except Exception as e:
                print(f"Error updating plots: {e}")
        
        # Update plots
        self.fig.clear()
        
        for i in range(60):
            ax = self.fig.add_subplot(6, 10, i+1)
            
            if len(self.channel_data[i]) > 0:
                data = np.array(list(self.channel_data[i]))
                
                # Convert to voltage (assuming 0.0087 mV per unit)
                data_voltage = data * 0.0087
                
                # Detect spikes
                std = np.std(data_voltage) if np.std(data_voltage) > 0 else 1
                threshold = self.spike_threshold * std
                spikes = np.where(np.abs(data_voltage) > threshold)[0]
                
                # Plot
                ax.plot(data_voltage, 'b-', linewidth=0.5)
                if len(spikes) > 0:
                    ax.plot(spikes, data_voltage[spikes], 'r.', markersize=4)
                
                ax.axhline(y=threshold, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(y=-threshold, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
            
            ax.set_title(f"Ch{i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-20, 20]) # Adjust limits appropriately
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Schedule next update
        self.root.after(100, self.update_plots)
    
    def on_closing(self):
        """Clean up on window close"""
        self.mea.disconnect()
        self.stimulator.disconnect()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = RealtimeGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
