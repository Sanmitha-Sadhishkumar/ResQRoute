import pandas as pd
import matplotlib.pyplot as plt

def fire_params_test(file_path):  # Change this to your actual file path
    # Step 1: Read the CSV file and skip the first row
    data = pd.read_csv(file_path, skiprows=1)  # Skips the first row with units
    
    # Step 2: Inspect and clean the data
    # Convert columns to numeric if they are in scientific notation
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing or irrelevant data
    data = data.dropna()

    # Step 3: Plot multiple parameters over time

    # Create subplots for different parameters
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot HRR (Heat Release Rate)
    axs[0].plot(data['Time'], data['HRR'], label='HRR (Heat Release Rate)', color='red')
    axs[0].set_title('Heat Release Rate (HRR) Over Time')
    axs[0].set_ylabel('HRR (kW)')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Q_RADI (Radiative Heat)
    axs[1].plot(data['Time'], data['Q_RADI'], label='Q_RADI (Radiative Heat)', color='blue')
    axs[1].set_title('Radiative Heat (Q_RADI) Over Time')
    axs[1].set_ylabel('Q_RADI (kW)')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Q_CONV (Convective Heat)
    axs[2].plot(data['Time'], data['Q_CONV'], label='Q_CONV (Convective Heat)', color='green')
    axs[2].set_title('Convective Heat (Q_CONV) Over Time')
    axs[2].set_ylabel('Q_CONV (kW)')
    axs[2].grid(True)
    axs[2].legend()

    # Plot MLR_AIR (Mass Flow Rate of Air)
    axs[3].plot(data['Time'], data['MLR_AIR'], label='MLR_AIR (Mass Flow Rate of Air)', color='purple')
    axs[3].set_title('Mass Flow Rate of Air (MLR_AIR) Over Time')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('MLR_AIR (kg/s)')
    axs[3].grid(True)
    axs[3].legend()

    # Adjust layout for better spacing between plots
    plt.tight_layout()

    # Show the plots
    plt.savefig('../static/show/fire_params.png')

    # Step 4: Analyze specific data points (for example, find the maximum HRR and its time)
    max_hrr = data['HRR'].max()
    max_time = data.loc[data['HRR'].idxmax(), 'Time']
    print(f"Maximum HRR: {max_hrr} kW at Time: {max_time} seconds")

    # Step 5: Save the cleaned data (optional)
    output_file_path = "cleaned_hrr_data.csv"
    data.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}")