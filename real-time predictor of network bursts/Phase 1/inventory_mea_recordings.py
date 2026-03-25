import os
import scipy.io

def inventory_and_convert_mea_recordings(root_folder, output_root):
    recordings = []

    # Walk through ALL subdirectories
    for root, dirs, files in os.walk(root_folder):

        # Compute relative path (to mirror structure)
        rel_path = os.path.relpath(root, root_folder)
        output_dir = os.path.join(output_root, rel_path)

        # Create matching output folder
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            if file.endswith('.mat'):
                input_path = os.path.join(root, file)
                output_file = os.path.join(output_dir, file.replace('.mat', '.csv'))

                print(f"Processing: {input_path}")

                recordings.append(input_path)

                # Load .mat file
                mat_data = scipy.io.loadmat(input_path)

                # Extract spike data
                spiketime_array = mat_data.get('spiketime')

                if spiketime_array is not None:
                    with open(output_file, 'w') as f:
                        f.write('spiketime,electrodes\n')

                        for row in spiketime_array:
                            f.write(f"{row[0]},{int(row[1])}\n")

                else:
                    print(f"'spiketime' not found in {file}")

    return recordings


# ROOT input 
root_folder = r"C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Cntr"

# OUTPUT root 
output_root = r"C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Output"

mea_recordings = inventory_and_convert_mea_recordings(root_folder, output_root)

print("\nFinished.")
print(f"Total files processed: {len(mea_recordings)}")