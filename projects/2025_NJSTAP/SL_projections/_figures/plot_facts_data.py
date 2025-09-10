import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import date

import glob
import os
import warnings

class SeaLevelProjection:
    def __init__(self, station,ssp, folder, data_opt, customize_plot_opt):
        self.folder = folder if isinstance(folder, list) else [folder]
        self.station = station
        self.station_name = self.get_station_name(station,data_opt)
        self.ssp = ssp
        self.customize_plot_opt = customize_plot_opt
        self.data_opt = data_opt
        self.colors = self.define_colors()
 
    
    def get_station_name(self, station, data_opt):
        region = data_opt.get('region')
        if region and region.lower() == "global":
            return "N/A"

        STATION_NAMES = {
            0: "PHILADELPHIA",
            1: "CAPE_MAY",
            2: "ATLANTIC_CITY",
            3: "SANDY_HOOK",
            4: "NEW_YORK"
        }
        return STATION_NAMES.get(station, f"Unknown_Station_{station}")


    def define_colors(self):
        return {
            'ssp119': np.array([0, 173, 207]) / 255,
            'ssp126': np.array([23, 60, 102]) / 255,
            'ssp245': np.array([247, 148, 32]) / 255,
            'ssp370': np.array([231, 29, 37]) / 255,
            'ssp585': np.array([149, 27, 30]) / 255
        }


    def project_SL(self):
        if len(self.folder) == 1:
            self._plot_single_subplot(self.data_opt, self.customize_plot_opt)
        else:
            self._plot_multiple_subplots(self.data_opt, self.customize_plot_opt)


    def _plot_single_subplot(self, data_opt, customize_plot_opt):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.4, hspace=0.2)
        self._plot_subplot(ax, data_opt, customize_plot_opt)
        plt.show()


    def _plot_multiple_subplots(self, data_opt, customize_plot_opt):
        #.@ improve logic
        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.2)
        self._plot_subplot(axes[0], data_opt, customize_plot_opt)
        self._plot_subplot(axes[1], data_opt, customize_plot_opt)
        # self._plot_subplot(axes[2], data_opt, customize_plot_opt)
        axes[2].axis('off')  # Turn off the third subplot axis
        plt.show()





    def _get_first_matching_file(self, folder_path, region, data_subfolder, ssp_value, file_name,file_type):
        import glob
        import os
        import warnings

        search_pattern = f"{folder_path}/{region}/{data_subfolder}/{ssp_value}/*{file_name}*{file_type}*.nc"
        matched_files = sorted(glob.glob(search_pattern))

        if not matched_files:
            raise FileNotFoundError(f"No files found matching: {search_pattern}")

        # If only one file, return it directly (regardless of suffix)
        if len(matched_files) == 1:
            return matched_files[0]

        # More than one file: prefer those ending with 'values.nc'
        filtered = [f for f in matched_files if f.endswith("values.nc")]

        if filtered:
            warnings.warn(
                f"Multiple files found for {ssp_value}. Using : {os.path.basename(filtered[0])}\n"
                f"All matches: {[os.path.basename(f) for f in matched_files]}"
            )
            return filtered[0]
        else:
            warnings.warn(
                f"Multiple files found for {ssp_value}, but none end with 'values.nc'. Using: {os.path.basename(matched_files[0])}\n"
                f"All matches: {[os.path.basename(f) for f in matched_files]}"
            )
            return matched_files[0]


    def _get_ssp_data_paths(self, data_opt):
        folder_path = data_opt.get('folder_path')
        region = data_opt.get('region')
        data_subfolder = data_opt.get('data_subfolder')
        file_name = data_opt.get('file_name')
        file_type = data_opt.get('file_type')
        ssp_folder_loop = [
            self._get_first_matching_file(folder_path, region, data_subfolder, ssp_value, file_name, file_type)
            for ssp_value in self.ssp
        ]

        return ssp_folder_loop



    def _load_nc_data(self, file_path):
        try:
            return xr.open_dataset(file_path, engine='netcdf4')
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    

    def _calculate_percentiles(self, dataset, data_var, unit):
        percentile = dataset['quantiles'].values * 100
        idx = np.where(percentile == 50)[0]  
        idx1 = np.where(percentile == 17)[0]  
        idx2 = np.where(percentile == 83)[0]  
        

        if unit == 'm':
            slc = dataset[data_var].isel(locations=self.station).values / 1000
        elif unit == 'cm':
            slc = dataset[data_var].isel(locations=self.station).values / 100
        elif unit == 'mm':
            slc = dataset[data_var].isel(locations=self.station).values
        else:
            raise ValueError(f"Invalid unit: {unit}. Supported units are 'm', 'cm', 'mm'.")

        time = dataset['years'].values

        return slc, time, idx, idx1, idx2



    def _plot_subplot(self, ax, data_opt, customize_plot_opt):
        ssp_folder_loop = self._get_ssp_data_paths(data_opt)
        yrST = data_opt.get('yrST', 2020)
        yrEN = data_opt.get('yrEN', 2150)
        data_var = data_opt.get('data_var')
        unit = data_opt.get('unit')

        lines, labels = [], []
        for i, (f_p, ssp_value) in enumerate(zip(ssp_folder_loop, self.ssp)):
            # Load NetCDF data
            dataset = self._load_nc_data(f_p)
            # Calculate percentiles and unit-converted sea level change data
            slc, time, idx, idx1, idx2 = self._calculate_percentiles(dataset, data_var, unit)
            # Determine the time index range for plotting
            idx_yr = np.where((time >= yrST) & (time <= yrEN))[0]
            # Plot shaded area for SSP 126 and SSP 585
            if ssp_value == 'ssp126' or ssp_value == 'ssp585':
                ax.fill_between(time[idx_yr], slc[idx1, idx_yr].reshape(-1),
                                slc[idx2, idx_yr].reshape(-1), color=self.colors[ssp_value], alpha=0.2)
            # Plot the median (50th percentile)
            line, = ax.plot(time[idx_yr], slc[idx, idx_yr].reshape(-1), color=self.colors[ssp_value])
            labels.append(f'{ssp_value[:3].upper()}{ssp_value[3]}-{ssp_value[4]}.{ssp_value[5]}')
            lines.append(line)

        # Customize the plot using additional options
        self._customize_plot(ax, lines, labels, customize_plot_opt,data_opt)


    def _customize_plot(self, ax, lines, labels, customize_plot_opt,data_opt):
        
        title = f"{' '.join(data_opt.get('data_subfolder', '').split('_')[:2])} {' - '} {data_opt.get('region', '')[:1].upper()}MSL Projections"

        
        ylab = f"{data_opt.get('region', '')[0].upper()}MSL rise (m; rel 1995-2014)"
        x_min = customize_plot_opt.get('x_min')
        x_max = customize_plot_opt.get('x_max')
        y_min = customize_plot_opt.get('y_min')
        y_max = customize_plot_opt.get('y_max')
        x_ticks = customize_plot_opt.get('x_ticks')
        y_ticks = customize_plot_opt.get('y_ticks')
        FS = customize_plot_opt.get('fontsize', 10)

        ax.set_xlabel('Year', fontsize=FS + (0.25 * FS))
        ax.set_ylabel(ylab, fontsize=FS + (0.25 * FS))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=FS, rotation=45)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=FS)
        ax.legend(lines, labels, loc='upper left', fontsize=FS)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

        ax.tick_params(direction='in', length=3.5, width=1, axis='both', top=True, right=True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)

        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.legend(lines + [ax.fill_between([], [], [], color='gray', alpha=0.2)], labels + ['Shading: 17-83 percentile'], loc='upper left', fontsize=FS + (0.1 * FS))
        ax.set_title(f'{title} {" "}  (station : {self.station_name})', fontsize=FS + (0.5 * FS))                  


    def save_plot(self, fig, base_filename="Fig1.example"):
        today = date.today().strftime('%Y-%m-%d')
        figure_name = f"{base_filename}_{today}.pdf"
        fig.savefig(figure_name, format='pdf'); print(f"Plot saved as {figure_name}")