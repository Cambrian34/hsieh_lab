"""
Real-Time MEA Recorder GUI
Records MEA or simulated data for a specified time span and saves it to disk.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os
from queue import Queue
import time

# MEA Device references
try:
    import clr
    dll_path = r"C:\Users\omi727\OneDrive - University of Texas at San Antonio\Desktop\LAb\hsieh_lab\Phase 2\McsUsbNet\x64\McsUsbNet.dll"
    if os.path.exists(dll_path):
        clr.AddReference(dll_path)
        from Mcs.Usb import CMcsUsbListNet, CMeaDeviceNet, McsBusTypeEnumNet, DeviceEnumNet, DataModeEnumNet, SampleSizeNet, SampleDstSizeNet
        from System import *
        MEA_AVAILABLE = True
    else:
        print(f"DLL not found at {dll_path}, running in simulation mode.")
        MEA_AVAILABLE = False
except Exception as e:
    print(f"CLR import failed: {e}\nRunning in simulation mode.")
    MEA_AVAILABLE = False


class MEADataAcquisition:
    """Handles MEA recording and simulated data generation."""
    def __init__(self, data_queue, num_channels=60):
        self.data_queue = data_queue
        self.num_channels = num_channels
        self.device = None
        self.is_recording = False
        self.sampling_rate = 25000
        self.callback_threshold = self.sampling_rate // 20  # 50ms windows
        self.simulation_thread = None
        self.simulation_active = False

    def on_channel_data(self, x, cbHandle, numSamples):
        try:
            data_arrays, _ = self.device.ChannelBlock.ReadAsFrameArrayI32(
                0, 0, self.callback_threshold, Int32(0)
            )
            data_array = np.asarray(data_arrays, dtype=np.int32)
            if not self.data_queue.full():
                self.data_queue.put(data_array)
        except Exception as e:
            print(f"Error in callback: {e}")

    def connect(self):
        if not MEA_AVAILABLE:
            return False
        try:
            device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
            if device_list.Count == 0:
                return False

            self.device = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS)
            self.device.ChannelDataEvent += self.on_channel_data
            self.device.Connect(device_list.GetUsbListEntry(0))

            self.device.SetSamplerate(self.sampling_rate, 1, 0)
            self.device.SetVoltageRangeByIndex(0, 0)
            self.device.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
            self.device.SetNumberOfChannels(0)
            self.device.SetNumberOfAnalogChannels(self.num_channels, 0, 0, 0, 0)
            self.device.EnableDigitalIn(Boolean(False), UInt32(0))
            self.device.EnableChecksum(False, 0)

            block = self.device.GetChannelsInBlock(0)
            m_channels = block // 2
            self.device.ChannelBlock.SetSelectedData(
                m_channels, self.sampling_rate, self.callback_threshold,
                SampleSizeNet.SampleSize32Signed, SampleDstSizeNet.SampleDstSize32, block
            )
            return True
        except Exception as e:
            print(f"MEA connection error: {e}")
            return False

    def _simulation_loop(self):
        self.simulation_active = True
        num_channels = self.num_channels
        chunk_size = self.callback_threshold
        while self.simulation_active:
            fake_data = np.random.randint(-2000, 2000, size=(chunk_size, num_channels), dtype=np.int32)
            for _ in range(np.random.randint(0, 4)):
                ch = np.random.randint(0, num_channels)
                start = np.random.randint(0, chunk_size - 10)
                amp = np.random.randint(8000, 20000) * (1 if np.random.rand() > 0.5 else -1)
                fake_data[start:start + 2, ch] = amp
                fake_data[start + 2:start + 5, ch] = -amp // 2
            if not self.data_queue.full():
                self.data_queue.put(fake_data)
            time.sleep(chunk_size / self.sampling_rate)
        self.simulation_active = False

    def start(self):
        if MEA_AVAILABLE and self.device:
            try:
                self.device.StartDacq()
                self.is_recording = True
                return True
            except Exception as e:
                print(f"Failed to start MEA acquisition: {e}")
                return False

        if self.simulation_thread is None or not self.simulation_thread.is_alive():
            self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.simulation_thread.start()
            self.is_recording = True
            return True

        return False

    def stop(self):
        if MEA_AVAILABLE and self.device and self.is_recording:
            try:
                self.device.StopDacq()
            except Exception:
                pass

        self.simulation_active = False
        self.is_recording = False

    def disconnect(self):
        self.stop()
        if MEA_AVAILABLE and self.device:
            try:
                self.device.Disconnect()
            except Exception:
                pass
            self.device = None


class RecorderGUI:
    """GUI for recording MEA data for a defined duration."""
    def __init__(self, root):
        self.root = root
        self.root.title("MEA Recorder")
        self.root.geometry("1400x820")

        self.data_queue = Queue(maxsize=10)
        self.mea = MEADataAcquisition(self.data_queue)
        self.recording = False
        self.record_data = None
        self.record_write_index = 0
        self.total_samples = 0
        self.target_samples = 0
        self.record_filename = None

        self.buffer_size = 2500
        self.num_channels = 60
        self.channel_data = np.zeros((self.num_channels, self.buffer_size), dtype=np.int32)
        self.write_index = 0
        self.display_decimation = 4
        self.spike_threshold = 3.0

        self.plot_initialized = False

        self.create_widgets()
        self.create_plot_lines()
        self.update_plots()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.connect_button = ttk.Button(control_frame, text="Connect MEA", command=self.connect_device)
        self.connect_button.pack(side=tk.LEFT, padx=4)

        self.status_label = ttk.Label(control_frame, text="Status: disconnected")
        self.status_label.pack(side=tk.LEFT, padx=10)

        ttk.Label(control_frame, text="Record time (s):").pack(side=tk.LEFT, padx=4)
        self.duration_var = tk.StringVar(value="5")
        self.duration_entry = ttk.Entry(control_frame, textvariable=self.duration_var, width=6)
        self.duration_entry.pack(side=tk.LEFT, padx=4)

        ttk.Label(control_frame, text="File:").pack(side=tk.LEFT, padx=4)
        self.filename_var = tk.StringVar(value="recording.npz")
        self.filename_entry = ttk.Entry(control_frame, textvariable=self.filename_var, width=24)
        self.filename_entry.pack(side=tk.LEFT, padx=4)

        ttk.Button(control_frame, text="Browse...", command=self.select_file).pack(side=tk.LEFT, padx=4)
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=8)

        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=8)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_plot_lines(self):
        self.axes = []
        self.lines = []
        self.spike_lines = []
        self.threshold_lines_pos = []
        self.threshold_lines_neg = []

        for i in range(self.num_channels):
            ax = self.fig.add_subplot(6, 10, i + 1)
            line, = ax.plot([], [], 'b-', linewidth=0.5)
            spike_line, = ax.plot([], [], 'r.', markersize=4)
            thresh_pos, = ax.plot([], [], 'g--', linewidth=0.5, alpha=0.5)
            thresh_neg, = ax.plot([], [], 'g--', linewidth=0.5, alpha=0.5)

            ax.set_title(f"Ch{i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([-20, 20])
            ax.set_xlim(0, self.buffer_size)

            self.axes.append(ax)
            self.lines.append(line)
            self.spike_lines.append(spike_line)
            self.threshold_lines_pos.append(thresh_pos)
            self.threshold_lines_neg.append(thresh_neg)

        self.fig.tight_layout()
        self.canvas.draw()
        self.plot_initialized = True

    def connect_device(self):
        connected = self.mea.connect()
        if connected:
            self.status_label.config(text="Status: MEA connected")
            self.connect_button.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="Status: simulation mode")
            self.connect_button.config(state=tk.DISABLED)
            self.mea.start()

    def select_file(self):
        path = filedialog.asksaveasfilename(
            title="Save recording as...",
            defaultextension='.npz',
            filetypes=[('NumPy compressed', '*.npz'), ('NumPy', '*.npy'), ('CSV', '*.csv')]
        )
        if path:
            self.filename_var.set(path)

    def toggle_recording(self):
        if not self.recording:
            try:
                duration = float(self.duration_var.get())
                if duration <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid duration", "Please enter a positive recording duration in seconds.")
                return

            filename = self.filename_var.get().strip()
            if not filename:
                messagebox.showerror("Invalid file", "Please choose a valid filename.")
                return

            self.start_recording(duration, filename)
        else:
            self.stop_recording(save=False)

    def start_recording(self, duration_seconds, filename):
        self.record_filename = filename
        self.target_samples = int(duration_seconds * self.mea.sampling_rate)
        self.record_data = np.zeros((self.target_samples, self.num_channels), dtype=np.int32)
        self.record_write_index = 0
        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.progress_label.config(text=f"Recording: 0 / {self.target_samples} samples")
        self.mea.start()

    def stop_recording(self, save=True):
        self.recording = False
        self.mea.stop()
        self.record_button.config(text="Start Recording")
        if save:
            self.save_recording()

    def save_recording(self):
        if self.record_data is None:
            return
        path = self.record_filename
        try:
            if path.lower().endswith('.npz'):
                np.savez_compressed(path, data=self.record_data[:self.record_write_index], sampling_rate=self.mea.sampling_rate)
            elif path.lower().endswith('.npy'):
                np.save(path, self.record_data[:self.record_write_index])
            else:
                np.savetxt(path, self.record_data[:self.record_write_index], delimiter=',', fmt='%d')
            self.progress_label.config(text=f"Saved {self.record_write_index} samples to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save recording: {e}")

    def update_plots(self):
        data_processed = False
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                samples = data.shape[0]
                end = self.write_index + samples
                if end <= self.buffer_size:
                    self.channel_data[:, self.write_index:end] = data.T
                else:
                    split = self.buffer_size - self.write_index
                    self.channel_data[:, self.write_index:] = data[:split].T
                    self.channel_data[:, :samples - split] = data[split:].T
                self.write_index = (self.write_index + samples) % self.buffer_size

                if self.recording and self.record_data is not None and self.record_write_index < self.target_samples:
                    remaining = self.target_samples - self.record_write_index
                    to_write = min(samples, remaining)
                    self.record_data[self.record_write_index:self.record_write_index + to_write] = data[:to_write]
                    self.record_write_index += to_write
                    if self.record_write_index >= self.target_samples:
                        self.stop_recording(save=True)
                data_processed = True
            except Exception as e:
                print(f"Error processing queue: {e}")

        if data_processed:
            x = np.arange(self.buffer_size)[::self.display_decimation]
            for i in range(self.num_channels):
                data_voltage = self.channel_data[i].astype(np.float32) * 0.0087
                display_voltage = data_voltage[::self.display_decimation]
                self.lines[i].set_data(x, display_voltage)
                recent_window = data_voltage[-500:]
                std = np.std(recent_window) if recent_window.size else 1.0
                threshold = self.spike_threshold * std
                spike_mask = np.abs(data_voltage) > threshold
                spike_indices = np.where(spike_mask)[0]
                spike_values = data_voltage[spike_indices]
                self.spike_lines[i].set_data(spike_indices, spike_values)
                self.threshold_lines_pos[i].set_data(x, np.full_like(x, threshold))
                self.threshold_lines_neg[i].set_data(x, np.full_like(x, -threshold))

            if self.recording:
                self.progress_label.config(text=f"Recording: {self.record_write_index} / {self.target_samples} samples")

            self.canvas.draw_idle()

        self.root.after(200, self.update_plots)

    def on_closing(self):
        self.mea.disconnect()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RecorderGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
