import h5py as h5
import numpy as np
import tifffile
import nibabel as nib
import os
from skimage import exposure


def read_hdf5(h5path, res, chan, lvl=None):
    """Reads an HDF5 file and extracts image data.

    Args:
        h5path (str): Path to the HDF5 file.
        res (str): Resolution key in the HDF5 file.
        chan (str): Channel key in the HDF5 file.
        lvl (int, optional): Specific level to extract (default: None, extract all).

    Returns:
        np.ndarray: Extracted image data.
    """
    with h5.File(h5path, 'r') as f:
        if lvl is None:
            img = f['t00000'][chan][res]['cells'][:,:,:].astype(np.uint16)
        else:
            img = f['t00000'][chan][res]['cells'][lvl:lvl+1,:,:].astype(np.uint16)
    return img


def read_tiff(file_path):
    """Reads a TIFF file into a NumPy array."""
    return tifffile.imread(file_path)


def ReadNPY(npy_path):
    loaded_dict = np.load(npy_path, allow_pickle=True).item()
    img = []
    for i in range(len(list(loaded_dict.keys()))):
        img.append(1-np.array(list(loaded_dict[i].values()))[0][0])
    mask_v = np.stack(img, dtype = np.int64)
    mask = np.moveaxis((1 - mask_v), 0, 2).astype(np.uint8)
    return mask


def read_niigz(file_path, level=None):
    """
    Read specific levels of a .nii.gz file to avoid overloading memory.
    
    Parameters:
    file_path (str): Path to the .nii.gz file.
    level (int or slice, optional): Specific level or slice to read. Default is None (read full data lazily).
    
    Returns:
    numpy.ndarray: The data of the specified level or slice.
    """
    img = nib.load(file_path)
    proxy = img.dataobj  # Lazy loading proxy

    # Get shape and determine the shortest axis
    shape = proxy.shape
    shortest_axis = np.argmin(shape)

    if level is not None:
        if shortest_axis == 0:
            data = proxy[level, :, :]
        elif shortest_axis == 1:
            data = proxy[:, level, :]
        elif shortest_axis == 2:
            data = proxy[:, :, level]
    else:
        # Load full data lazily
        data = proxy[:]
        # Move the shortest axis to the 3rd position
        axes_order = [i for i in range(len(shape)) if i != shortest_axis] + [shortest_axis]
        data = np.moveaxis(data, axes_order, np.arange(len(shape)))

    return data


def writetiff(img, output_path):
    """Writes a 16-bit TIFF file with minimal compression."""
    print("Saving TIFF...")
    with tifffile.TiffWriter(output_path) as tif:
        for i in range(len(img)):
            tif.write(img[i], contiguous=True)


def save_niigz(numpy_arr, output_path):
    """Saves a NumPy array as a .nii.gz file."""
    stitched_img = nib.Nifti1Image(numpy_arr, np.eye(4))
    nib.save(stitched_img, output_path)


def chunk(h5path, overlap = 0.25, blocksize = 1024, ftype = ".nii.gz", res="0", nlvl = ""):
    """
    Params: 
    - h5path
    - savepath
    - overlap_percentage (can be 0)
    - ftype: ".nii.gz" or ".tiff"
    - nlvl: how many levels from the top
    - res: resolution, use "0"
    - nblk (for chunking by specifying number of blocks)
    - blocksize (for chunking by specifying blocksize)
    """
    savepath = os.path.dirname(h5path) + os.sep + "nnunet"

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    chan = [("s01", "0001"),
            ("s00", "0000" )]

    with h5.File(h5path, 'r') as f:
        img_shape = f['t00000'][chan[0][0]][res]['cells'].shape
    f.close()

    x_num_blocks = int((img_shape[1] - overlap * blocksize) // ((1 - overlap) * blocksize))
    y_num_blocks = int((img_shape[2] - overlap * blocksize) // ((1 - overlap) * blocksize))

    # Loop through each block
    for ch in chan:
        for i in range(x_num_blocks):
            for j in range(y_num_blocks):
                # Calculate start and end indices for x and y dimensions with overlap
                x_start = int(i * blocksize * (1-overlap))
                x_end = int(x_start + blocksize)
                y_start = int(j * blocksize * (1-overlap))
                y_end = int(y_start + blocksize)

                with h5.File(h5path, 'r') as f:
                    if nlvl == "":
                        img = f['t00000'][ch[0]][res]['cells'][:, x_start:x_end, y_start:y_end].astype(np.uint16)
                    else:
                        img = f['t00000'][ch[0]][res]['cells'][:nlvl, x_start:x_end, y_start:y_end].astype(np.uint16)
                    if ftype == ".nii.gz":
                        img = np.moveaxis(img,0,2)
                f.close()

                # Prepare filename for the block
                # fname = os.path.basename(os.path.dirname(h5path)) + f"_blk_{i}_{j}_" + ch[1] + ftype
                fname = h5path.split("-23_")[1].split("_well")[0] + f"_blk_{i}_{j}_" + ch[1] + ftype
                fpath = savepath + os.sep + fname

                # # Print the indices for verification (you can remove this if not needed)
                if ftype == ".nii.gz":
                    img = exposure.rescale_intensity(img,
                            in_range=(np.min(img), np.percentile(img,99.99)), 
                            out_range=np.uint8)
                    img = img.astype(np.uint8)
                    save_niigz(img, fpath)

                else:
                    writetiff(img, fpath)

                print(x_start, x_end, y_start, y_end)


def Blend(savepath, output_path, blocksize = 1024, overlap_percentage = 0.25):
    # Find all block files in the directory
    block_files = [f for f in os.listdir(savepath) if f.endswith('.nii.gz')]

    # Extract the image shape from one of the blocks
    example_block = read_niigz(os.path.join(savepath, block_files[0]))
    depth = example_block.shape[2]

    # Calculate the number of blocks in x and y directions
    xy_values = [(int(block_file.split('_')[2]), int(block_file.split('_')[3].split('.')[0])) for block_file in block_files]
    x_blocks = max(x for x,y in xy_values) + 1
    y_blocks = max(y for x,y in xy_values) + 1

    # Calculate the dimensions of the final stitched volume
    x_dim = int(blocksize + (x_blocks - 1) * blocksize * (1 - overlap_percentage))
    y_dim = int(blocksize + (y_blocks - 1) * blocksize * (1 - overlap_percentage))

    # Create an empty volume to hold the stitched image
    stitched_volume = np.zeros((x_dim, y_dim, depth), dtype=np.uint16)

    # Iterate over each block file and place it in the stitched volume
    for block_file in block_files:
        i = int(block_file.split('_')[2])
        j = int(block_file.split('_')[3].split(".nii.gz")[0])

        # Calculate start and end indices for x and y dimensions
        x_start = int(i * blocksize * (1 - overlap_percentage))
        x_end = x_start + blocksize
        y_start = int(j * blocksize * (1 - overlap_percentage))
        y_end = y_start + blocksize

        block_data = read_niigz(os.path.join(savepath, block_file))

        # Apply max blending in the overlapping region
        stitched_volume[x_start:x_end, y_start:y_end] = np.maximum(
            stitched_volume[x_start:x_end, y_start:y_end], block_data)

    return stitched_volume
