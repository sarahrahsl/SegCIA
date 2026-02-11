import numpy as np
from scipy.ndimage import distance_transform_edt, zoom, label, measurements
from pathlib import Path
from .data_handling import read_niigz, read_tiff
from skimage.measure import regionprops, marching_cubes
from skimage.measure import label as sklabel
from scipy.spatial import ConvexHull


def compute_distance_maps(nerve_mask, pixelno_adj, pixelno_dis, tolerance=1):
    distance_map = distance_transform_edt(~nerve_mask)
    dilated_adj = distance_map <= pixelno_adj
    dilated_dis = distance_map <= pixelno_dis
    region_adj = dilated_adj & ~nerve_mask
    region_dis = dilated_dis & ~dilated_adj
    surface_adj = (distance_map >= pixelno_adj - tolerance) & (distance_map <= pixelno_adj + tolerance)
    surface_dis = (distance_map >= pixelno_dis - tolerance) & (distance_map <= pixelno_dis + tolerance)
    return region_adj, region_dis, surface_adj, surface_dis


def crop3D(nerve_mask, cancer_mask, z_levels, zoom_factor = (0.25, 0.25, 0.25)):
    """
    nerve_mask: 3D numpy array of downsampled, cropped nerve mask [x,y,z]
    cancer_mask: 3D numpy array of downsampled, UNcropped cancer mask, shape (X, Y, Z)
    z_levels: tuple of start and end z levels to crop
    """

    # Crop Z levels
    start, end = z_levels
    start = round(start//4)
    end = round(end//4)

    if end > 160:
        cancer_mask_cropped = cancer_mask[:, :, start:160]
    else:
        cancer_mask_cropped = cancer_mask[:, :, start:end]
    nerve_mask_cropped = nerve_mask

    # Ensure dimensions match
    print("nerve shape", nerve_mask_cropped.shape, "cancer shape", cancer_mask_cropped.shape)
    if not nerve_mask_cropped.shape[2] == cancer_mask_cropped.shape[2]:
        if nerve_mask_cropped.shape[2] > cancer_mask_cropped.shape[2]:
            nerve_mask_cropped = nerve_mask_cropped[:, :, :cancer_mask_cropped.shape[2]]
        if nerve_mask_cropped.shape[2] < cancer_mask_cropped.shape[2]:
            cancer_mask_cropped = cancer_mask_cropped[:, :, :nerve_mask_cropped.shape[2]]

    if not nerve_mask_cropped.shape[1] == cancer_mask_cropped.shape[1]:
        if nerve_mask_cropped.shape[1] > cancer_mask_cropped.shape[1]:
            nerve_mask_cropped = nerve_mask_cropped[:, :cancer_mask_cropped.shape[1], :]
        if nerve_mask_cropped.shape[1] < cancer_mask_cropped.shape[1]:
            cancer_mask_cropped = cancer_mask_cropped[:, :nerve_mask_cropped.shape[1], :]

    if not nerve_mask_cropped.shape[0] == cancer_mask_cropped.shape[0]:
        if nerve_mask_cropped.shape[0] > cancer_mask_cropped.shape[0]:
            nerve_mask_cropped = nerve_mask_cropped[:cancer_mask_cropped.shape[0], :, :]
        if nerve_mask_cropped.shape[0] < cancer_mask_cropped.shape[0]:
            cancer_mask_cropped = cancer_mask_cropped[:nerve_mask_cropped.shape[0], :, :]
    
    # Ensure nerve mask is boolean and convert cancer mask from 0,255 to 0,1
    cancer_mask_cropped = (cancer_mask_cropped == 255).astype(bool)

    return nerve_mask_cropped, cancer_mask_cropped


def get_gradient_for_instance(volumes, gradients, mode = 'max'):
    """
    Returns the gradient corresponding to the instance with the maximum volume.

    Parameters:
        volumes (list or np.ndarray): List/array of volume values.
        gradients (list or np.ndarray): List/array of gradient values (same length, may include np.nan).

    Returns:
        float: Gradient value corresponding to the maximum volume (may be np.nan).
    """
    # Convert to numpy arrays but keep original lengths to allow mismatched inputs
    volumes_arr = np.array(volumes) if volumes is not None else np.array([])
    gradients_arr = np.array(gradients) if gradients is not None else np.array([])

    # If either is empty or lengths mismatch, try to align by trimming to the minimum length
    if volumes_arr.size == 0 or gradients_arr.size == 0:
        return np.nan

    if volumes_arr.size != gradients_arr.size:
        minlen = min(volumes_arr.size, gradients_arr.size)
        volumes_arr = volumes_arr[:minlen]
        gradients_arr = gradients_arr[:minlen]

    # If after trimming we still have no valid entries, return nan
    if volumes_arr.size == 0:
        return np.nan

    if mode == 'max':
        # nanargmax will raise if all-NaN; guard against that
        try:
            idx = np.nanargmax(volumes_arr)
        except ValueError:
            return np.nan
    elif mode == 'median':
        # Sort volumes and get index of the one closest to the median value
        try:
            sorted_indices = np.argsort(volumes_arr)
            median_idx = sorted_indices[len(volumes_arr) // 2]
            idx = median_idx
        except Exception:
            return np.nan
    else:
        return np.nan

    # If gradients_arr at idx is NaN or out of bounds, return nan
    try:
        val = gradients_arr[idx]
    except Exception:
        return np.nan
    return val


# Compute ratios
def safe_divide(a, b):
    try:
        if b is None:
            return np.nan
        # handle cases where a or b may be numpy arrays or scalars
        if isinstance(b, (int, float, np.floating, np.integer)):
            return a / b if b > 0 else np.nan
        # if b is array-like, perform elementwise safe divide
        b_arr = np.array(b)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(a, b_arr)
            if np.isscalar(res):
                return res if np.isfinite(res) else np.nan
            return np.where(np.isfinite(res), res, np.nan)
    except Exception:
        return np.nan


def Calculate_invasion_feature_2D(cancer_path, mask_path, z_levels,
                                  sample_name, 
                                  pixelno_adj=20, pixelno_dis=40, 
                                  sliceno=None):
    
    # Read the data
    seg_mask = read_niigz(mask_path)  # Shape: (X, Y, Z)
    cancer_mask = read_niigz(cancer_path)  # Shape: (X, Y, Z)
    seg_mask_cropped, cancer_mask_cropped = crop3D(seg_mask, cancer_mask, z_levels)
    nerve_mask_cropped = seg_mask_cropped.astype(bool)

    # Define all the keys for the stats dictionary

    stats_keys = [
        "sample_name",
        "mean_cancer_Area_adj", "med_cancer_Area_adj", "min_cancer_Area_adj", "max_cancer_Area_adj", "sd_cancer_Area_adj",
        "mean_percentageArea_adj", "med_percentageArea_adj", "min_percentageArea_adj", "max_percentageArea_adj", "sd_percentageArea_adj",
        "Area_gradient_of_max_cancer_adj", "Area_percentage_gradient_of_max_cancer_adj", "Area_invasion_of_max_cancer_adj", "Area_percentage_invasion_of_max_cancer_adj",
        "Area_gradient_of_median_cancer_adj", "Area_percentage_gradient_of_median_cancer_adj", "Area_invasion_of_median_cancer_adj", "Area_percentage_invasion_of_median_cancer_adj",
        "mean_Area_gradient", "med_Area_gradient", "min_Area_gradient", "max_Area_gradient", "sd_Area_gradient",
        "mean_Area_invasion", "med_Area_invasion", "min_Area_invasion", "max_Area_invasion", "sd_Area_invasion",
        "mean_percentageArea_gradient", "med_percentageArea_gradient", "min_percentageArea_gradient", "max_percentageArea_gradient", "sd_percentageArea_gradient",
        "mean_percentageArea_invasion", "med_percentageArea_invasion", "min_percentageArea_invasion", "max_percentageArea_invasion", "sd_percentageArea_invasion",
        "mean_Peri_gradient", "med_Peri_gradient", "min_Peri_gradient", "max_Peri_gradient", "sd_Peri_gradient",
        "mean_Peri_invasion", "med_Peri_invasion", "min_Peri_invasion", "max_Peri_invasion", "sd_Peri_invasion",
        "mean_percentagePeri_gradient", "med_percentagePeri_gradient", "min_percentagePeri_gradient", "max_percentagePeri_gradient", "sd_percentagePeri_gradient",
        "mean_percentagePeri_invasion", "med_percentagePeri_invasion", "min_percentagePeri_invasion", "max_percentagePeri_invasion", "sd_percentagePeri_invasion"
    ]
    prefix1, prefix2 = "Area", "Peri"
    
    # Initialize stats dictionary with NaN values
    if np.sum(nerve_mask_cropped) == 0:
        stats = {key: np.nan for key in stats_keys}
        stats["sample_name"] = sample_name
    
    else:
        # Empty lists to store data for each instance
        individual_cancer_volumes_adj = []
        individual_cancer_surface_adj = []
        volumes_percentages_adj = []
        surfaces_percentages_adj = []
        gradients = []
        percentage_gradients = []
        invasions = []
        percentage_invasions = []
        surface_gradients = []
        surface_invasions = []
        surface_percentage_gradients = []
        surface_percentage_invasions = []
        
        # Process slice by slice
        if sliceno is not None:
            slices = sliceno
        else:
            slices = range(nerve_mask_cropped.shape[2])

        for i in slices:
            print(i)
            nerve = nerve_mask_cropped[:, :, i]
            cancer = cancer_mask_cropped[:, :, i]
            labeled_nerves, num_labels = label(nerve) # Label individual nerves on a 2D mask
            # print("number of nerve fragment:", num_labels)

            if num_labels > 0:

                # Loop over each labeled nerve fragment
                for label_id in range(1, num_labels + 1):
                    # Isolate nerve fragment and create corresponding annulus
                    nerve_fragment = labeled_nerves == label_id
                    # if not np.sum(nerve_fragment) < 90000:
                    region_fragment_adj, region_fragment_dis, surface_fragment_adj, surface_fragment_dis = \
                                            compute_distance_maps(nerve_fragment, pixelno_adj, pixelno_dis)
                    
                    # Calculate cancer volumes within regions for this nerve fragment
                    cancer_volume_fragment_adj = np.sum(cancer[region_fragment_adj])
                    cancer_volume_fragment_dis = np.sum(cancer[region_fragment_dis])
                    cancer_surface_fragment_adj = np.sum(cancer[surface_fragment_adj])
                    cancer_surface_fragment_dis = np.sum(cancer[surface_fragment_dis])
                    volume_adj_sum = np.sum(region_fragment_adj)
                    volume_dis_sum = np.sum(region_fragment_dis)
                    surface_adj_sum = np.sum(surface_fragment_adj)
                    surface_dis_sum = np.sum(surface_fragment_dis)
                    percentage_volume_fragment_adj = cancer_volume_fragment_adj / volume_adj_sum if volume_adj_sum > 0 else None
                    percentage_volume_fragment_dis = cancer_volume_fragment_dis / volume_dis_sum if volume_dis_sum > 0 else None
                    percentage_surface_fragment_adj = cancer_surface_fragment_adj / surface_adj_sum if surface_adj_sum > 0 else None
                    percentage_surface_fragment_dis = cancer_surface_fragment_dis / surface_dis_sum if surface_dis_sum > 0 else None

                    # Append results for cancer content
                    individual_cancer_volumes_adj.append(cancer_volume_fragment_adj)
                    individual_cancer_surface_adj.append(cancer_surface_fragment_adj)
                    volumes_percentages_adj.append(percentage_volume_fragment_adj)
                    surfaces_percentages_adj.append(percentage_surface_fragment_adj)
                    # Append results for gradients and invasions
                    gradients.append(safe_divide(cancer_volume_fragment_dis, cancer_volume_fragment_adj))
                    percentage_gradients.append(safe_divide(percentage_volume_fragment_dis, percentage_volume_fragment_adj))
                    invasions.append(safe_divide(percentage_volume_fragment_adj, percentage_volume_fragment_dis))
                    percentage_invasions.append(safe_divide(percentage_volume_fragment_adj, percentage_volume_fragment_dis))
                    surface_gradients.append(safe_divide(cancer_surface_fragment_dis, cancer_surface_fragment_adj))
                    surface_invasions.append(safe_divide(cancer_surface_fragment_adj, cancer_surface_fragment_dis))
                    surface_percentage_gradients.append(safe_divide(percentage_surface_fragment_dis, percentage_surface_fragment_adj))
                    surface_percentage_invasions.append(safe_divide(percentage_surface_fragment_adj, percentage_surface_fragment_dis))


        print("num volumes", len(individual_cancer_volumes_adj))
        print("num gradients", len(gradients))

        stats = {} 
        stats["sample_name"] = sample_name
        stats[f"mean_cancer_{prefix1}_adj"]= np.nanmean(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan
        stats[f"med_cancer_{prefix1}_adj"]= np.nanmedian(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan
        stats[f"min_cancer_{prefix1}_adj"]= np.nanmin(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan
        stats[f"max_cancer_{prefix1}_adj"]= np.nanmax(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan
        stats[f"sd_cancer_{prefix1}_adj"]= np.std(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan

        stats[f"mean_percentage{prefix1}_adj"]= np.nanmean(volumes_percentages_adj) if volumes_percentages_adj else np.nan
        stats[f"med_percentage{prefix1}_adj"]= np.nanmedian(volumes_percentages_adj) if volumes_percentages_adj else np.nan
        stats[f"min_percentage{prefix1}_adj"]= np.nanmin(volumes_percentages_adj) if volumes_percentages_adj else np.nan
        stats[f"max_percentage{prefix1}_adj"]= np.nanmax(volumes_percentages_adj) if volumes_percentages_adj else np.nan
        stats[f"sd_percentage{prefix1}_adj"]= np.std(volumes_percentages_adj) if volumes_percentages_adj else np.nan

        # Just to find the corresponding gradient information for the instance of nerve that has the highest cancer content
        stats[f"{prefix1}_gradient_of_max_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, gradients, mode='max')
        stats[f"{prefix1}_percentage_gradient_of_max_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, percentage_gradients, mode='max')
        stats[f"{prefix1}_invasion_of_max_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, invasions, mode='max')
        stats[f"{prefix1}_percentage_invasion_of_max_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, percentage_invasions, mode='max')
        stats[f"{prefix1}_gradient_of_median_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, gradients, mode='median')
        stats[f"{prefix1}_percentage_gradient_of_median_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, percentage_gradients, mode='median')
        stats[f"{prefix1}_invasion_of_median_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, invasions, mode='median')
        stats[f"{prefix1}_percentage_invasion_of_median_cancer_adj"] = get_gradient_for_instance(volumes_percentages_adj, percentage_invasions, mode='median')

        stats[f"mean_{prefix1}_gradient"]= np.nanmean(gradients) if gradients else np.nan
        stats[f"med_{prefix1}_gradient"]= np.nanmedian(gradients) if gradients else np.nan
        stats[f"min_{prefix1}_gradient"]= np.nanmin(gradients) if gradients else np.nan
        stats[f"max_{prefix1}_gradient"]= np.nanmax(gradients) if gradients else np.nan
        stats[f"sd_{prefix1}_gradient"]= np.std(gradients) if gradients else np.nan

        stats[f"mean_{prefix1}_invasion"]= np.nanmean(invasions) if invasions else np.nan
        stats[f"med_{prefix1}_invasion"]= np.nanmedian(invasions) if invasions else np.nan
        stats[f"min_{prefix1}_invasion"]= np.nanmin(invasions) if invasions else np.nan
        stats[f"max_{prefix1}_invasion"]= np.nanmax(invasions) if invasions else np.nan
        stats[f"sd_{prefix1}_invasion"]= np.std(invasions) if invasions else np.nan

        stats[f"mean_percentage{prefix1}_gradient"]= np.nanmean(percentage_gradients) if percentage_gradients else np.nan
        stats[f"med_percentage{prefix1}_gradient"]= np.nanmedian(percentage_gradients) if percentage_gradients else np.nan
        stats[f"min_percentage{prefix1}_gradient"]= np.nanmin(percentage_gradients) if percentage_gradients else np.nan
        stats[f"max_percentage{prefix1}_gradient"]= np.nanmax(percentage_gradients) if percentage_gradients else np.nan
        stats[f"sd_percentage{prefix1}_gradient"]= np.std(percentage_gradients) if percentage_gradients else np.nan

        stats[f"mean_percentage{prefix1}_invasion"]= np.nanmean(percentage_invasions) if percentage_invasions else np.nan
        stats[f"med_percentage{prefix1}_invasion"]= np.nanmedian(percentage_invasions) if percentage_invasions else np.nan
        stats[f"min_percentage{prefix1}_invasion"]= np.nanmin(percentage_invasions) if percentage_invasions else np.nan
        stats[f"max_percentage{prefix1}_invasion"]= np.nanmax(percentage_invasions) if percentage_invasions else np.nan
        stats[f"sd_percentage{prefix1}_invasion"]= np.std(percentage_invasions) if percentage_invasions else np.nan

        stats[f"mean_{prefix2}_gradient"]= np.nanmean(surface_gradients) if surface_gradients else np.nan
        stats[f"med_{prefix2}_gradient"]= np.nanmedian(surface_gradients) if surface_gradients else np.nan
        stats[f"min_{prefix2}_gradient"]= np.nanmin(surface_gradients) if surface_gradients else np.nan
        stats[f"max_{prefix2}_gradient"]= np.nanmax(surface_gradients) if surface_gradients else np.nan
        stats[f"sd_{prefix2}_gradient"]= np.std(surface_gradients) if surface_gradients else np.nan

        stats[f"mean_{prefix2}_invasion"]= np.nanmean(surface_invasions) if surface_invasions else np.nan
        stats[f"med_{prefix2}_invasion"]= np.nanmedian(surface_invasions) if surface_invasions else np.nan
        stats[f"min_{prefix2}_invasion"]= np.nanmin(surface_invasions) if surface_invasions else np.nan
        stats[f"max_{prefix2}_invasion"]= np.nanmax(surface_invasions) if surface_invasions else np.nan
        stats[f"sd_{prefix2}_invasion"]= np.std(surface_invasions) if surface_invasions else np.nan

        stats[f"mean_percentage{prefix2}_gradient"]= np.nanmean(surface_percentage_gradients) if surface_percentage_gradients else np.nan
        stats[f"med_percentage{prefix2}_gradient"]=  np.nanmedian(surface_percentage_gradients) if surface_percentage_gradients else np.nan
        stats[f"min_percentage{prefix2}_gradient"]=  np.nanmin(surface_percentage_gradients) if surface_percentage_gradients else np.nan
        stats[f"max_percentage{prefix2}_gradient"]=  np.nanmax(surface_percentage_gradients) if surface_percentage_gradients else np.nan
        stats[f"sd_percentage{prefix2}_gradient"]=  np.std(surface_percentage_gradients) if surface_percentage_gradients else np.nan

        stats[f"mean_percentage{prefix2}_invasion"]= np.nanmean(surface_percentage_invasions) if surface_percentage_invasions else np.nan
        stats[f"med_percentage{prefix2}_invasion"]=  np.nanmedian(surface_percentage_invasions) if surface_percentage_invasions else np.nan
        stats[f"min_percentage{prefix2}_invasion"]=  np.nanmin(surface_percentage_invasions) if surface_percentage_invasions else np.nan
        stats[f"max_percentage{prefix2}_invasion"]=  np.nanmax(surface_percentage_invasions) if surface_percentage_invasions else np.nan
        stats[f"sd_percentage{prefix2}_invasion"]=  np.std(surface_percentage_invasions) if surface_percentage_invasions else np.nan

    return stats



def Annular_chunk_analysis(
    cancer_mask, 
    nerve_mask, 
    sample_name, 
    chunk_size=(200, 200, 200), 
    pixelno_adj=20, 
    pixelno_dis=40, 
    stride=0.5
):
    assert cancer_mask.shape == nerve_mask.shape, "Shape mismatch between masks"
    x_max, y_max, z_max = nerve_mask.shape
    dx, dy, _ = chunk_size  # Full z-range

    # Stride as overlap fraction (0.5 = 50%)
    step_x = max(1, int(dx * stride))
    step_y = max(1, int(dy * stride))

    # Lists to collect per-chunk values
    perc_ca_adj_all = []
    perc_ca_dis_all = []
    perc_grad_all = []
    perc_inva_all = []
    chunk_data = []

    def safe_stats(arr, prefix):
        return {
            f"{prefix}_max": np.nanmax(arr) if arr else np.nan,
            f"{prefix}_min": np.nanmin(arr) if arr else np.nan,
            f"{prefix}_mean": np.nanmean(arr) if arr else np.nan,
            f"{prefix}_median": np.nanmedian(arr) if arr else np.nan,
            f"{prefix}_std": np.nanstd(arr) if arr else np.nan
        }

    for x in range(0, x_max - dx + 1, step_x):
        for y in range(0, y_max - dy + 1, step_y):
            # Full z-range
            nerve_chunk = nerve_mask[x:x+dx, y:y+dy, :]
            cancer_chunk = cancer_mask[x:x+dx, y:y+dy, :]

            if nerve_chunk.shape != (dx, dy, z_max):
                continue

            nerve_bin = nerve_chunk.astype(bool)
            if np.sum(nerve_bin) == 0:
                continue

            # Define annular regions
            adj_region, dis_region, _, _ = compute_distance_maps(nerve_bin, pixelno_adj, pixelno_dis)

            # Volumes
            adj_vol = np.sum(adj_region)
            dis_vol = np.sum(dis_region)
            ca_vol_adj = np.sum(cancer_chunk[adj_region])
            ca_vol_dis = np.sum(cancer_chunk[dis_region])
            perc_ca_adj = ca_vol_adj / adj_vol if adj_vol > 0 else np.nan
            perc_ca_dis = ca_vol_dis / dis_vol if dis_vol > 0 else np.nan
            perc_grad = perc_ca_dis / perc_ca_adj if perc_ca_adj > 0 else np.nan
            perc_inva = perc_ca_adj / perc_ca_dis if perc_ca_dis > 0 else np.nan

            # Store metrics
            perc_ca_adj_all.append(perc_ca_adj)
            perc_ca_dis_all.append(perc_ca_dis)
            perc_grad_all.append(perc_grad)
            perc_inva_all.append(perc_inva)

            chunk_data.append({
                "x": x,
                "y": y,
                "perc_ca_adj": perc_ca_adj,
                "perc_ca_dis": perc_ca_dis,
                "perc_gradient": perc_grad,
                "perc_invasion": perc_inva
            })

    # Identify max and median % cancer in adjacent
    stats = {"sample_name": sample_name}
    if perc_ca_adj_all:
        sorted_idx = np.argsort(perc_ca_adj_all)
        max_idx = sorted_idx[-1]
        med_idx = sorted_idx[len(sorted_idx) // 2]

        max_chunk = chunk_data[max_idx]
        med_chunk = chunk_data[med_idx]

        stats.update({
            "top_chunk_perc_ca_adj": max_chunk["perc_ca_adj"],
            "top_chunk_perc_gradient": max_chunk["perc_gradient"],
            "top_chunk_perc_invasion": max_chunk["perc_invasion"],
            "median_chunk_perc_ca_adj": med_chunk["perc_ca_adj"],
            "median_chunk_perc_gradient": med_chunk["perc_gradient"],
            "median_chunk_perc_invasion": med_chunk["perc_invasion"]
        })
    else:
        stats.update({
            "top_chunk_perc_ca_adj": np.nan,
            "top_chunk_perc_gradient": np.nan,
            "top_chunk_perc_invasion": np.nan,
            "median_chunk_perc_ca_adj": np.nan,
            "median_chunk_perc_gradient": np.nan,
            "median_chunk_perc_invasion": np.nan
        })

    # Flatten overall stats
    stats.update(safe_stats(perc_ca_adj_all, "perc_ca_adj"))
    stats.update(safe_stats(perc_ca_dis_all, "perc_ca_dis"))
    stats.update(safe_stats(perc_grad_all, "perc_gradient"))
    stats.update(safe_stats(perc_inva_all, "perc_invasion"))

    return stats, chunk_data
