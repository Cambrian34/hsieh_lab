import os
import scipy.io

# Function to inventory MEA recordings and convert .mat files

def inventory_and_convert_mea_recordings(folder_path, output_folder):
    recordings = []
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parse through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mat'):
                recordings.append(os.path.join(root, file))
                # Load .mat file
                mat_data = scipy.io.loadmat(os.path.join(root, file))
                # Save converted data to the new folder 
                output_file = os.path.join(output_folder, file.replace('.mat', '.csv'))
            
                #check data to see what if the header is there 
                #print(f"Headers in {file}:")
                #for key in mat_data.keys():
                
                #    if not key.startswith('__'):
                #
                #         print(f"  - {key}")

                
                #Each file contains a double array spiketime, where the first column is the spikes time (from 0 
                #to 1800000 msec), the second is the #electrodes which detected a spikes (from 1 to 60). 
                spiketime_array = mat_data.get('spiketime')
                if spiketime_array is not None:
                    with open(output_file, 'w') as f: #two headers spiketime and electrodes
                        f.write('spiketime,electrodes\n')  # Write the header
                        # Since the spiketime is an array opf two then the first column data goes under spiketime and
                        #the second value column the array the data goes under electrodes
                        for row in spiketime_array:
                            f.write(f"{row[0]},{int(row[1])}\n")
                    
                    

                    


    return recordings

# folder path to search and output folder
folder_path = 'C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Cntr\\420_I15918' 
output_folder = 'C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Output' 

# Get the list of recordings and convert them
mea_recordings = inventory_and_convert_mea_recordings(folder_path, output_folder)

if mea_recordings:
    print("Available 24–48h MEA recordings (DIV 20+):")
    for recording in mea_recordings:
        print(recording)
else:
    print("No MEA recordings found.")
