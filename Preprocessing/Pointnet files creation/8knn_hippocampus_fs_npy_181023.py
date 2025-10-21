import re
import torch
import os
import pandas as pd
import numpy as np
from torch_cluster import knn_graph
import os.path as osp
#from torch_geometric.data import InMemoryDataset, Data

from torch_geometric.data import Dataset, Data

from os.path import exists

class BrainScanDataset(Dataset):
    
    data_url = "https://raw.githubusercontent.com/smweis/Silcton_MRI/master/public_data/DataAnalysisWith90Participants_Jupyter.csv"
    
    def __init__(self, root, region_name, transform=None, pre_transform=None, pre_filter=None):
        print (region_name)
        self.region_name = region_name
        super(BrainScanDataset, self).__init__(root, transform, pre_transform, pre_filter)
        print ("AFTER SUPER")
        # Load processed data
        #self.data, self.slices = torch.load(self.processed_paths[0])
        print ("LINE 30")
        #self.region_name = region_name
        print (self.region_name)

    @property
    def raw_file_names(self):
        subjects = pd.read_csv(r'/blue/stevenweisberg/ashishkumarsahoo/hippocampAI/participants.tsv',sep='\t',header=0)
        subjs = subjects.participant_id
        regions_dir = "/blue/stevenweisberg/ashishkumarsahoo/difumo/Data"
        
        self.file_list = ["{}/{}/{}_hippocampusmasked_brain_standardized_noneg_normalized.npy".format(regions_dir, self.region_name, subj) for subj in subjs]
        #self.file_list = ["{}/{}/sub-1002/sub-1002_difumo_dim_combined_with_orig_intensity.csv".format(regions_dir, self.region_name)]
        return self.file_list
    
    @property
    def processed_file_names(self):
        subjects = pd.read_csv(r'/blue/stevenweisberg/ashishkumarsahoo/hippocampAI/participants.tsv',sep='\t',header=0)
        subjs = subjects.participant_id
        self.output_list = ["data_{}.pt".format(subj) for subj in subjs]
        return self.output_list
        
    def process(self):
        data_list = []
        
        file_exists = exists('/blue/stevenweisberg/ashishkumarsahoo/difumo/Data/processed/data_ity.pt')
        if file_exists == True:
            return
        else:
            csv = pd.read_csv(self.data_url)

            #extra
            #self.column_list = [f'dim_{n}' for n in range(0,64)]
            #idx = 0
            for npy_file in self.raw_file_names:
                print(npy_file)
                data = np.load(npy_file)

                # Create a mask for values greater than 0
                mask = data > 0
                
                # Apply the mask to extract voxels with values greater than 0
                extracted_data = data[mask]
                
                indices = np.where(mask)
                # Separate the indices into x, y, and z coordinates
                x_indices, y_indices, z_indices = indices
                
                # Create the position array using the index arrays
                pos_array = np.vstack(indices).T
                
                # # Get the shape of the array
                # array_shape = data.shape

                # # Generate index arrays for each dimension
                # indices = np.indices(array_shape)

                # # Extract individual index arrays
                # index_arrays = []
                # for dim_indices in indices:
                #     index_array = dim_indices.flatten()
                #     index_arrays.append(index_array)

                # # Create the position array using the index arrays
                # pos_array = np.vstack(index_arrays).T
                
                #pos_array = np.array(data[['x','y','z']])
                
                # Flatten the NumPy array to obtain the values
                extracted_data_values = data[indices]
                col_array = extracted_data_values.flatten()
                
                #col_array = np.asarray(data['intensity'])
                #col_array = np.asarray(data[self.column_var])
                #pcd_array = np.column_stack((pos_array, col_array))

                x = torch.from_numpy(col_array)
                pos = torch.from_numpy(pos_array)
                edge_index = knn_graph(pos, k=8)

                subject = re.search("sub-(\d+)_", npy_file).group(1)
                subject = int(subject)
                model_building_total = csv[csv.ID == subject].Model_Building_Total.values[0]
                model_building_total = torch.tensor(model_building_total, dtype=torch.float)

                data = Data(x=x, num_nodes=pos_array.shape[0],
                            edge_index=edge_index,
                            y=model_building_total, pos=pos)
                data_list.append(data)
                torch.save(data, osp.join(save_dir, f'8knn_hippocampus_fs_{subject}.pt'))
                del data
                torch.cuda.empty_cache()
                
                #torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                #idx += 1
            #torch.save(data_list, osp.join(self.processed_dir, 'data_3dim_65feat_k8_pos_0208.pt'))
            #torch.save(data_list, osp.join(self.processed_dir, 'data_3dim_1feat_4knn_2908.pt'))

    def len(self):
        return len(self.processed_file_names)
    
ply_dir = "/blue/stevenweisberg/ashishkumarsahoo/difumo/Data"
region_name = "freesurfer_121023/hippocampus_standardized_normalized"
save_dir = "/blue/stevenweisberg/share/hippocampAI/8knn_hippocampus_fs_181023"
brainDataset = BrainScanDataset(ply_dir, region_name)

print('no of participants is ', len(brainDataset))

total_size = len(brainDataset)
