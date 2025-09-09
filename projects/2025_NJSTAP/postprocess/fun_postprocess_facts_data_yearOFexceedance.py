import numpy as np
import xarray as xr
import glob
import os
import shutil
import fnmatch
import re
from netCDF4 import Dataset
import warnings
#
from pathlib import Path
import time
#

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Function block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_directory_structure(notebook_dir,folder_name, subfolder_name=None, subsubfolder_name=None):

    def recreate_folder(path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        return path

    paths = [recreate_folder(os.path.join(notebook_dir, folder_name))]
    
    if subfolder_name:
        paths.append(recreate_folder(os.path.join(paths[-1], subfolder_name)))
    if subsubfolder_name:
        paths.append(recreate_folder(os.path.join(paths[-1], subsubfolder_name)))
    
    return tuple(paths)
# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
region = "local"

common_files = [
    f'lws.ssp.landwaterstorage_{region}sl.nc',
    f'ocean.tlm.sterodynamics_{region}sl.nc',
    *(['k14vlm.kopp14.verticallandmotion_localsl.nc'] if region == "local" else [])
]

workflow_components = {
    "wf_1e": ['emuGrIS.emulandice.GrIS', 		  'emuAIS.emulandice.AIS', 			  'emuglaciers.emulandice.glaciers'],
    "wf_1f": ['GrIS1f.FittedISMIP.GrIS_GIS', 	  'ar5AIS.ipccar5.icesheets_AIS', 	  'ar5glaciers.ipccar5.glaciers'],
    "wf_2e": ['emuGrIS.emulandice.GrIS', 		  'larmip.larmip.AIS', 				  'emuglaciers.emulandice.glaciers'],
    "wf_2f": ['GrIS1f.FittedISMIP.GrIS_GIS', 	  'larmip.larmip.AIS', 				  'ar5glaciers.ipccar5.glaciers'],
    "wf_3e": ['emuGrIS.emulandice.GrIS', 		  'deconto21.deconto21.AIS_AIS', 	  'emuglaciers.emulandice.glaciers'],
    "wf_3f": ['GrIS1f.FittedISMIP.GrIS_GIS', 	  'deconto21.deconto21.AIS_AIS', 	  'ar5glaciers.ipccar5.glaciers'],
    "wf_4":  ['bamber19.bamber19.icesheets_GIS',  'bamber19.bamber19.icesheets_AIS',  'ar5glaciers.ipccar5.glaciers'],
}

component 	= {key: [f'{item}_{region}sl.nc' for item in items] + common_files for key, items in workflow_components.items()}

total 		= {key: [f'total.workflow.{key.replace("_", "")}.{region}.nc'] for key in workflow_components}

# ^^^


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Year of Exceedance Calculation 
#++++++++++++++++++++++++++++++++++++++++

def sample_from_quantiles(qvals, q, nsamps):
    pool = np.interp(np.linspace(0, 1, nsamps + 2)[1:(nsamps + 1)], q, qvals)
    np.random.shuffle(pool)
    return pool


def YearsOfExceedance(sample, years, heights):
    # Determine years when specific SL heights are exceeded.
    sample_max = np.maximum.accumulate(sample)   
    exyears = np.full_like(heights, 9999, dtype=np.float64)            # start with all 9999
    valid = heights <= np.max(sample_max)                              # T/F boolean array
    exyears[valid] = np.interp(heights[valid], sample_max, years)      # interpolates the year when each height in heights[valid] is reached
    return np.around(exyears)  #  rounded to the nearest whole number.


def ExtractData(infile, outfile, heights, nsamps=5000):    
    
    # Quantiles to extract
    q = np.array([0.05, 0.17, 0.5, 0.83, 0.95])

    # Open the input netCDF file
    nc = Dataset(infile, 'r')
    
    # Extract data
    ncyears = nc.variables['years'][...]
    nclocalsl = nc.variables['sea_level_change'][...]
    ncquantiles = nc.variables['quantiles'][...]
    ncsiteids = nc.variables['locations'][...]
    nclat = nc.variables['lat'][...]
    nclon = nc.variables['lon'][...]
    nc.close()

    infile_string = ", ".join("/".join(Path(f).parts[-3:]) for f in [infile])

    # Store exceedance years
    exyears_q = []    
    
    # Process each site
    for j in range(len(ncsiteids)):
        samps = []

        for i in range(len(ncyears)):
            np.random.seed(8071)  # Ensures reproducibility
            samps.append(sample_from_quantiles(nclocalsl[:, i, j], ncquantiles, nsamps))   

        samps = np.array(samps) 
        
        """ YearsOfExceedance (MODIFIED function call) """ 
        this_exyears = np.apply_along_axis(YearsOfExceedance, 0, samps, years=ncyears, heights=heights)    #(0==> loop over columns ie over nsamps trajectories.)

        # Compute Qunatiles
        this_exyears_q = np.quantile(this_exyears, q, axis=1)
        
        """********************************************* 
        2301 :: Handel the 9999 data.
        9999 means: “Height is exceeded at an unknown time”
        *********************************************"""
        this_exyears_q = np.where(this_exyears_q == 9999,2301 if ncyears[-1] > 2100 else np.nan,this_exyears_q)

        exyears_q.append(this_exyears_q)


    # Convert to numpy array
    exyears_q = np.array(exyears_q)

    # Write output netCDF
    rootgrp = Dataset(outfile, "w", format="NETCDF4")

    # Create dimensions
    rootgrp.createDimension("quantiles", len(q))
    rootgrp.createDimension("heights", len(heights))
    rootgrp.createDimension("locations", len(ncsiteids))

    # Create variables
    q_var = rootgrp.createVariable("quantiles", "f4", ("quantiles",))
    height_var = rootgrp.createVariable("heights", "i2", ("heights",))
    site_var = rootgrp.createVariable("locations", "i4", ("locations",))
    lat_var = rootgrp.createVariable("lat", "f4", ("locations",))
    lon_var = rootgrp.createVariable("lon", "f4", ("locations",))
    exyearsq = rootgrp.createVariable( 
          "exceedance_years", "f4", 
          ("quantiles", "heights", "locations"), 
          zlib=True, complevel=4, fill_value=np.nan)
    exyearsq.coordinates = "lat lon"

    # Assign attributes
    rootgrp.description = "Years of Exceedance data"
    rootgrp.history = f" Created {time.ctime()}"
    rootgrp.source = f"Files Combined: {infile_string}"
    exyearsq.units = "mm"

    # Save data
    q_var[:] = q
    height_var[:] = heights
    site_var[:] = ncsiteids
    lat_var[:] = nclat
    lon_var[:] = nclon
    exyearsq[:, :, :] = np.transpose(exyears_q, (1, 2, 0))   

    # Close the output file
    rootgrp.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def postprocess(workflow,ssps,expFolder, folder2create, operation):    
    
    Workflow_path = create_directory_structure(os.getcwd(), folder2create)
    
    for indv_workflow in workflow:
        wf_path = create_directory_structure(Workflow_path[0], indv_workflow)
        
        for ssp in ssps:
    
            ssp_path = create_directory_structure(wf_path[0], ssp)
            sourcefile_path = f'{expFolder}/coupling.{ssp}/output'
    
            for wfcomp in component[indv_workflow] + total[indv_workflow]:
                if operation == "compute_YearsOfExceedance":
                            inFile  = f'{os.getcwd()}/2_workflow_quantiles/{"/".join(ssp_path[0].split("/")[-2:])}/coupling.{ssp}.{wfcomp}'
                            if not os.path.exists(inFile):
                                raise ValueError(f"File does not exist: {inFile}")
                            outFile = f"{ssp_path[0]}/coupling.{ssp}." + wfcomp.replace('.nc', '_xcdance.nc')
                            ## ==>mm calc
                            ## ExtractData(inFile, outFile, heights=np.arange(100, 5001, 100))
                            # ==>Feet calc
                            heights_ft = np.array([1, 2, 3, 4, 5, 6, 7])
                            heightF = heights_ft * 304.8  # now in mm
                            ExtractData(inFile, outFile, heights=heightF) 

                

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  p-box function block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def LoadInfiles(infiles, years):
    
    valid_varnames = ["exceedance_years"]

    ds_list = [xr.open_dataset(infile) for infile in infiles]
    varname = next((v for v in valid_varnames if v in ds_list[0]), None)
    if not varname:
        raise Exception(f"No valid variable name exists in the first file.")

    varunit = "mm"
    
    ids, lats, lons = [ds_list[0][val].values for val in ['locations', 'lat', 'lon']]
    qvar 			= np.round(ds_list[0]['quantiles'].values, 3)

    height_list = np.array([ds[varname].heights.values for ds in ds_list])
    heights = height_list[0]
    localsl_q = np.array([ds[varname].sel(heights=heights).values for ds in ds_list])

    """honor pbox limits for calcultion.""" 
    if years[-1] == 2100:
        localsl_q = np.where(localsl_q > years[-1], np.nan, localsl_q) ## Remove values beyond 2100 for pb1e,2e computation. This trims wf4.
    elif years[-1] == 2300:
        localsl_q = localsl_q # Do nothing, nan exceedance already marked as 2301
    else:
        warnings.warn(f"Unexpected final year: {years[-1]}. Check dataset consistency.")

    [ds.close() for ds in ds_list]
        
    return localsl_q, varname, varunit, ids, lats, lons, qvar, heights


def create_pbox(infiles, outfile, pyear_start=2020, pyear_end=2300, pyear_step=10):
    
    """ pyear_end is defined as pyear_end = 2300 if wf_num in ['1f', '2f'] else 2100 
        in the process_indv_Pbox.  """
    years = np.arange(pyear_start, pyear_end + 1, pyear_step)
    component_data, varname, varunit, ids, lats, lons, qvar, heights = LoadInfiles(infiles, years)
    infile_string = ", ".join("/".join(Path(f).parts[-3:]) for f in infiles)
    
    median_idx_array = np.flatnonzero(qvar == 0.5)
    median_idx = int(median_idx_array[0])

	# Create p-box bounding 
    pbox = np.full(component_data.shape[1:], np.nan)
    pbox[median_idx,:,:]    = np.nanmean(component_data[:,median_idx,:,:], axis=0)
    pbox[:median_idx,:,:]   = np.nanmin(component_data[:,:median_idx,:,:], axis=0)
    pbox[median_idx+1:,:,:] = np.nanmax(component_data[:,median_idx+1:,:,:], axis=0)
    
    ds = xr.Dataset({varname: (("quantiles", "heights", "locations"), pbox)},
                    coords={"quantiles": qvar,"heights": heights,"locations": ids,
                            "lat": ("locations", lats), "lon": ("locations", lons)})
    
    ds[varname].attrs.clear()
    ds[varname].attrs.update({"units": varunit})
    
    ds.attrs.clear()
    ds.attrs.update({"Description": "Pbox generated from Years of Exceedance",
                     "History"    : f"Created {time.ctime()}",
                     "Source"     : f"Files Combined: {infile_string}",
                     "Author"     : "Praveen Kumar <praveen.kumar@rutgers.edu>",})

    # save the file
    ds.to_netcdf(outfile, format="NETCDF4", encoding={varname: {"zlib": True, "complevel": 4, "dtype": "float64", "_FillValue": np.nan}})



# Populate pbox folder
def create_pbox_from_component(base_paths, patterns, outfile, pyear_end):
    infiles = []
    for base_path, pattern in zip(base_paths, patterns):
        matched_files = glob.glob(os.path.join(base_path, f"*{pattern}*"))
        if not matched_files:
            raise FileNotFoundError(f"No files found for pattern: {pattern} in {base_path}")
        infiles.append(matched_files[0])

    create_pbox(infiles, outfile, pyear_start=2020, pyear_end=pyear_end, pyear_step=10)
    os.remove(infiles[0])


def process_indv_Pbox(indv_Pbox, ssp, ssp_path):
    wf_num = indv_Pbox.split('_')[1]
    
    folder_name = ssp_path.split(os.sep)[-3]
    mapping = {
        "3_pbox":               "2_workflow_quantiles",
        "3_pbox_rates":         "2_workflow_quantiles_rates",
        "3_pbox_xcdance":       "2_YearsOfExceedance"       
    }
    if folder_name in mapping:
        workflow_folder = [mapping[folder_name]][0]
    else:
        raise ValueError(f"Invalid folder name: {folder_name}")
    
    # Dynamically pick dir
    src_dir = f"{os.getcwd()}/{workflow_folder}/wf_{wf_num}/{ssp}"
    shutil.copytree(src_dir, ssp_path, dirs_exist_ok=True)

    """
    Define P-Box outer time bounds
    """
    pyear_end = 2300 if wf_num in ['1f', '2f'] else 2100

    # AIS Component
    if indv_Pbox == 'pb_1e':
        base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_2e/{ssp}"]
        patterns = ["emulandice.AIS", "larmip.AIS"]
    
    elif indv_Pbox == 'pb_1f':
        base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_2f/{ssp}"]
        patterns = ["ipccar5.icesheets_AIS", "larmip.AIS"]
    
    elif indv_Pbox == 'pb_2e':
        base_paths = [ssp_path,
            f"{os.getcwd()}/{workflow_folder}/wf_1e/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_3e/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"]
        patterns = ["larmip.AIS", 
                    "emulandice.AIS", 
                    "deconto21.AIS", 
                    "bamber19.icesheets_AIS"]
    
    elif indv_Pbox == 'pb_2f':
        base_paths = [ssp_path,
            f"{os.getcwd()}/{workflow_folder}/wf_1f/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_3f/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"
        ]
        patterns = ["larmip.AIS", 
                    "ipccar5.icesheets_AIS", 
                    "deconto21.AIS", 
                    "bamber19.icesheets_AIS"]

    create_pbox_from_component(base_paths, patterns, f"{ssp_path}/icesheets-pb{wf_num}-icesheets-{ssp}_AIS_localsl.nc", pyear_end)

    # GrIS & Glacier for pb_2e and pb_2f
    if indv_Pbox in ['pb_2e', 'pb_2f']:
        
        if indv_Pbox == 'pb_2e':
            base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"]
            patterns = ["emulandice.GrIS", "bamber19.icesheets_GIS"]
            create_pbox_from_component(base_paths, patterns, f"{ssp_path}/icesheets-pb{wf_num}-icesheets-{ssp}_GIS_localsl.nc", pyear_end)

            base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_1f/{ssp}"]
            patterns = ["emulandice.glaciers", "ipccar5.glaciers"]
            create_pbox_from_component(base_paths, patterns, f"{ssp_path}/glaciers-pb{wf_num}-glaciers-{ssp}_localsl.nc", pyear_end)

        elif indv_Pbox == 'pb_2f':
            base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"]
            patterns = ["GrIS1f.FittedISMIP.GrIS", "bamber19.icesheets_GIS"]
            create_pbox_from_component(base_paths, patterns, f"{ssp_path}/icesheets-pb{wf_num}-icesheets-{ssp}_GIS_localsl.nc", pyear_end)

    # Total Component
    if indv_Pbox == 'pb_1e':
        base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_2e/{ssp}"]
        patterns = ["total.workflow.wf1e", "total.workflow.wf2e"]
    elif indv_Pbox == 'pb_1f':
        base_paths = [ssp_path, f"{os.getcwd()}/{workflow_folder}/wf_2f/{ssp}"]
        patterns = ["total.workflow.wf1f", "total.workflow.wf2f"]
    elif indv_Pbox == 'pb_2e':
        base_paths = [
            ssp_path,
            f"{os.getcwd()}/{workflow_folder}/wf_1e/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_3e/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"
        ]
        patterns = ["total.workflow.wf2e", "total.workflow.wf1e", "total.workflow.wf3e", "total.workflow.wf4"]
    elif indv_Pbox == 'pb_2f':
        base_paths = [
            ssp_path,
            f"{os.getcwd()}/{workflow_folder}/wf_1f/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_3f/{ssp}",
            f"{os.getcwd()}/{workflow_folder}/wf_4/{ssp}"
        ]
        patterns = ["total.workflow.wf2f", "total.workflow.wf1f", "total.workflow.wf3f", "total.workflow.wf4"]

    create_pbox_from_component(base_paths, patterns, f"{ssp_path}/total-workflow.nc", pyear_end)


def process_pbox(pbox,outFolderName,ssps):
    pbox_path = create_directory_structure(os.getcwd(), outFolderName)[0]

    for indv_pbox in pbox:
        pb_path = create_directory_structure(pbox_path, indv_pbox)[0]
    
        for ssp in ssps:
            ssp_path = create_directory_structure(pb_path, ssp)[0]
            process_indv_Pbox(indv_pbox, ssp, ssp_path)



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CL fun
def GetScenarios(dir):

	# Get the scenario names from the available scenario directories
	# Ignore any hidden directories (i.e. .DS_Store)
	pb1e_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_1e")) if not re.search(r"^\.", x)]
	pb1f_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_1f")) if not re.search(r"^\.", x)]
	pb2e_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_2e")) if not re.search(r"^\.", x)]
	pb2f_scenarios = [x for x in os.listdir(os.path.join(dir, "pb_2f")) if not re.search(r"^\.", x)]

	# Find the overlapping scenarios
	med_scenarios = list(set(pb1e_scenarios) & set(pb1f_scenarios))
	low_scenarios = list(set(pb2e_scenarios) & set(pb2f_scenarios))

	# Return the overlapping scenarios
	med_scenarios.sort()
	low_scenarios.sort()
	return(med_scenarios, low_scenarios)



def GetFiles(dir):

	file_keys = ["glaciers", "landwaterstorage", "sterodynamics", "AIS", "GIS", "total", "verticallandmotion"]

	# Initialize list of matched file keys
	match_files = {}

	# Loop over the keys and find the associated files
	for this_key in file_keys:

		# Locate this file in the directory
		pattern = "*{}*.nc".format(this_key)
		this_file = fnmatch.filter(os.listdir(dir), pattern)

		# There should be only one match
		if len(this_file) == 1:

			# Append the match
			match_files[this_key] = os.path.join(dir, this_file[0])

		elif len(this_file) > 1:
			raise Exception("More than one file matched in {} for key {}".format(dir, this_key))

		else:

			match_files[this_key] = None

	# Return the dictionary of files
	return(match_files)



def MakeConfidenceFile(infile_e=None, infile_f=None, f_years=np.arange(2020,2301,10), outfile=None, is_rates=False):

	if infile_f is None and infile_e is None:
		return(1)

	# Variable names and attributes
	if is_rates:
		varname = "sea_level_change_rate"
	else:
		varname = "exceedance_years"

	# Open and subset the f file
	with xr.open_dataset(infile_f) as nc_f:
		nc_out = nc_f

	# Add the f file to the source list
	source_files = [infile_f]


	# If there's an e file, overlap it with the f file
	if infile_e is not None:
		with xr.open_dataset(infile_e) as nc_e:
			nc_out = nc_e.combine_first(nc_f)

		# Append the e file to the source file list
		source_files.append(infile_e)

	nc_missing_value = np.nan
      
    # Attributes for the output file
	nc_attrs = {"Description": "Confidence output generated from Years of Exceedance Pbox",
		        "History"    : f"Created {time.ctime()}",
		        "Source"     : "Files Combined: {}".format(",".join("/".join(Path(f).parts[-3:]) for f in source_files)),
                "Author"     : "Praveen Kumar <praveen.kumar@rutgers.edu>"}

	# Put the attributes onto the output file
	nc_out.attrs = nc_attrs

	# Write the output file
	nc_out.to_netcdf(outfile, encoding={varname: {"dtype": "float64", "zlib": True, "complevel":4, "_FillValue": np.nan}})

	return(None)



def GenerateConfidenceFiles(pboxdir, outdir):

	# Are we working with values or rates?
	is_rates = True if re.search(r"rates", pboxdir) is not None else False

	# Get the overlapping scenarios for each confidence level
	med_scenarios, low_scenarios = GetScenarios(pboxdir)

	# If these are rate pboxes...
	if is_rates:

		# Loop over the medium scenarios
		for this_scenario in med_scenarios:

			# Get the list of files for this scenario
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_1f", this_scenario))

			# Loop over the available components
			for this_key in pb1f_infiles.keys():

				# Define the output file name
				outpath = Path(os.path.join(outdir, "medium_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_medium_confidence_rates.nc".format(this_key, this_scenario))
				# Make the output file
				MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile, is_rates=is_rates)

		# Loop over the low scenarios
		for this_scenario in low_scenarios:

			# Get the list of files for this scenario
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_2f", this_scenario))

			# Loop over the available components
			for this_key in pb1f_infiles.keys():
				# Define the output file name
				outpath = Path(os.path.join(outdir, "low_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_low_confidence_rates.nc".format(this_key, this_scenario))

				MakeConfidenceFile(infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2301,10), outfile=outfile, is_rates=is_rates)
				

	else:
		# Loop over the medium scenarios
		for this_scenario in med_scenarios:

			# Get the list of files for this scenario
			pb1e_infiles = GetFiles(os.path.join(pboxdir, "pb_1e", this_scenario))
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_1f", this_scenario))

			# Loop over the available components
			for this_key in pb1e_infiles.keys():
				outpath = Path(os.path.join(outdir, "medium_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_medium_confidence_xcdance.nc".format(this_key, this_scenario))
				# Make the output file
				MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2151,10), outfile=outfile)

		# Loop over the low scenarios
		for this_scenario in low_scenarios:
			pb1e_infiles = GetFiles(os.path.join(pboxdir, "pb_2e", this_scenario))
			pb1f_infiles = GetFiles(os.path.join(pboxdir, "pb_2f", this_scenario))

			for this_key in pb1e_infiles.keys():
				outpath = Path(os.path.join(outdir, "low_confidence", this_scenario))
				Path.mkdir(outpath, parents=True, exist_ok=True)
				outfile = os.path.join(outpath, "{}_{}_low_confidence_xcdance.nc".format(this_key, this_scenario))
                # Make the output file
				MakeConfidenceFile(infile_e=pb1e_infiles[this_key], infile_f=pb1f_infiles[this_key], f_years=np.arange(2020,2301,10), outfile=outfile)

	return(None)
# ^^^